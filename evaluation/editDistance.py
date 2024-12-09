#!~/miniconda3/bin/python python3

# python3 ~/scripts/editDistance/editDistance_trans_reco.py ~/kaldi/egs/tme_song/test/data/ksong-test-500_hires/text ~/kaldi/egs/tme_song/test/exp/chain/cnntdnnf_6k/decode_mixed.uniq.1e-11_ksong-test-500/scoring_kaldi/penalty_0.0/8.txt ~/results/ksong-test-500-6k

import operator
import os
import re
import sys
from multiprocessing import Pool
from string import punctuation as puncen

import numpy as np
from zhon.hanzi import punctuation as puncch

# punctuations except < > ' etc.
puncs = '\\'.join(
    set(puncch + puncen + '▁' + '¿' + '¡') -
    {'\'', '<', '>', '@', '#', '%', '^', '&', '_'})
re_puncs = re.compile(fr'[{puncs}]+')
del puncs

# cjk
cjks = list(range(0x3400, 0x4DB5 + 1)) + list(range(
    0x4E00, 0x9FCB + 1)) + list(range(0x20000, 0x2A6D6 + 1)) + list(
        range(0x2A700, 0x2B734 + 1))
# thai
cjks += list(range(0x0E00, 0x0E7F + 1))
cjks = [chr(x) for x in cjks]
cjks = ''.join(cjks)
# add space for CJK
re_bound = re.compile(rf'(?<=[{cjks}])(?!\s)|(?<!\s)(?=[{cjks}])')
del cjks
# remove tags
re_tags = re.compile(r'\<[^\s]*?\>')


class actions:
    astart, amatch, asub, ains, adel = range(0, 5)


class edit_dist:

    def __init__(self,
                 fmatch=None,
                 cost={
                     'ins': 1,
                     'sub': 1,
                     'del': 1,
                     'match': 0
                 }):
        self.syms = {'ins': '^', 'del': '-', 'sub': '=>'}
        self.cost = cost
        if not fmatch:
            self.is_match = self.is_match_default
        else:
            self.is_match = fmatch

    @staticmethod
    def is_match_default(w1, w2):
        if w1 == w2:
            return True
        else:
            return False

    def edist(self, x, y, key=''):
        '''
        f(x) -> y
        think: x ~ ref     y ~ hyp
        numpy arrays are row major
        '''
        nx = len(x)
        ny = len(y)
        d = np.zeros((nx + 1, ny + 1), dtype=int)
        p = np.zeros((nx + 1, ny + 1), dtype=int)

        d[0, 1:] = [self.cost['ins'] * i for i in range(1, ny + 1)]
        d[1:, 0] = [self.cost['del'] * i for i in range(1, nx + 1)]
        p[0, 1:] = [actions.ains for i in range(1, ny + 1)]
        p[1:, 0] = [actions.adel for i in range(1, nx + 1)]

        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                choices = []
                if self.is_match(x[i - 1], y[j - 1]):
                    choices.append(
                        (d[i - 1, j - 1] + self.cost['match'], actions.amatch))
                else:
                    choices.append(
                        (d[i - 1, j - 1] + self.cost['sub'], actions.asub))
                choices.append((d[i - 1, j] + self.cost['del'], actions.adel))
                choices.append((d[i, j - 1] + self.cost['ins'], actions.ains))
                best_action = sorted(choices, key=operator.itemgetter(0))
                d[i, j] = best_action[0][0]
                p[i, j] = best_action[0][1]

        # backtrack
        alignment = [[], []]
        i = nx
        j = ny
        while i > 0 or j > 0:
            pp = p[i, j]
            if pp == actions.amatch:
                alignment[0].append(x[i - 1])
                alignment[1].append(y[j - 1])
                i -= 1
                j -= 1
            elif pp == actions.asub:
                alignment[0].append(x[i - 1])
                alignment[1].append(self.syms['sub'] + y[j - 1])
                i -= 1
                j -= 1
            elif pp == actions.ains:
                alignment[0].append(self.syms['ins'])
                alignment[1].append(y[j - 1])
                j -= 1
            elif pp == actions.adel:
                alignment[0].append(x[i - 1])
                alignment[1].append(self.syms['del'])
                i -= 1
            else:
                print(f'bad backpointer={pp}')
                i -= 1
                j -= 1
        alignment[0].reverse()
        alignment[1].reverse()
        a = list(zip(alignment[0], alignment[1]))

        return d[nx, ny], a


def ali_trans(ref, hyp, ali):
    res = []
    S, I, D, M = 0, 0, 0, 0
    for i in range(len(ali)):
        word0 = ali[i][0]
        word1 = ali[i][1]
        if word1.startswith('=>'):
            res.append('S')
            S += 1
        elif word0.startswith('^'):
            res.append('I')
            ref.insert(i, '**')
            I += 1
        elif word1.startswith('-'):
            res.append('D')
            hyp.insert(i, '**')
            D += 1
        else:
            res.append('M')
            M += 1
    return ref, hyp, res, S, I, D, M


def ali_convert(ali):
    res = []
    for pair in ali:
        word0 = pair[0]
        word1 = pair[1]
        if word1.startswith('=>'):
            res.append('S')
        elif word0.startswith('^'):
            res.append('I')
        elif word1.startswith('-'):
            res.append('D')
        else:
            res.append('M')
    return res


def edist(params):

    key, ref, hyp = params

    # remove punctuations and add spaces between CJKs
    ref = ref.lower()
    ref = re_tags.sub(' ', ref)
    ref = re_puncs.sub(' ', ref)
    ref = re_bound.sub(' ', ref)
    ref = ref.strip().split()

    hyp = hyp.lower()
    hyp = re_tags.sub(' ', hyp)
    hyp = re_puncs.sub(' ', hyp)
    hyp = re_bound.sub(' ', hyp)
    hyp = hyp.strip().split()

    e = edit_dist()
    dis, ali = e.edist(ref, hyp)

    num_tkn = len(ref)
    ref, hyp, ali, S, I, D, M = ali_trans(ref, hyp, ali)

    return key, ref, hyp, ali, num_tkn, dis, S, I, D, M


def read_textfile(textfile):
    dict_text = dict()
    for line in open(textfile, encoding='utf-8'):
        tmp = line.strip().split('\t')
        key = tmp[0]
        key = os.path.basename(key)
        key = os.path.splitext(key)[0]
        trans = tmp[1]
        dict_text[key] = trans
    return dict_text


if __name__ == '__main__':

    transfile = sys.argv[1]
    recofile = sys.argv[2]

    utt = 0
    total_tkn = 0
    total_dis = 0

    # read trans and reco
    dict_trans = read_textfile(transfile)
    dict_reco = read_textfile(recofile)
    items = [(key, trans, dict_reco[key]) for key, trans in dict_trans.items()]

    # compute edit distance
    with Pool(16) as pool:
        res = pool.map(edist, items)

    # count edist
    total_S, total_I, total_D, total_M = 0, 0, 0, 0
    for tmp in sorted(res):

        key, ref, hyp, ali, num_tkn, dis, S, I, D, M = tmp
        total_S += S
        total_I += I
        total_D += D
        total_M += M

        total_tkn += num_tkn
        total_dis += dis
        utt += 1
        err = dis / num_tkn
        print(f'{key}\t{dis}/{num_tkn}={err:.3f}')
        print(' '.join(hyp))
        print(' '.join(ref))
        print('  '.join(ali))

    res = f'All utts={utt}, Ins={total_I}, Del={total_D}, Sub={total_S},\nMatch={total_M}, WER={total_dis}/{total_tkn}={total_dis/total_tkn:.4f}'
    print(res)
