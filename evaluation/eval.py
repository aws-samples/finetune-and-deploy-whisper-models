import argparse
import datetime
import os
import os.path as osp
from multiprocessing import Pool
from pathlib import Path

import datasets
import torch
import tqdm
from datasets import Dataset
from editDistance import edist
from peft import PeftModel
from rich import print as rprint
from rich.table import Table
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.pt_utils import KeyDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Speech recognition and evaluation')
    parser.add_argument('--data-dirs',
                        nargs='+',
                        required=True,
                        help='List of data directories')
    parser.add_argument('--list-file', type=str, default='trans.txt')
    parser.add_argument(
        '--language',
        choices=['en', 'ms', 'id', 'zh'],
        required=True,
        help=('Choose a language: en (English), ms (Malay), id (Indonesian), '
              'or zh (Chinese)'))
    parser.add_argument('--model-id',
                        type=str,
                        default='openai/whisper-large-v3',
                        help='Model ID')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Batch size for inference')
    parser.add_argument('--chunk-length',
                        type=int,
                        default=60,
                        help='Input length for in each chunk')
    parser.add_argument('--log-path',
                        type=str,
                        default='./logs',
                        help='Path to the log directory')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--lora-path',
                        type=str,
                        default=None,
                        help='Path to Lora file')
    parser.add_argument('--prompt',
                        type=str,
                        default='',
                        help='pre-defined prompt')
    args = parser.parse_args()
    return args


def evaluate_pred(names, refs, preds, log_file=None):
    # compute edit distance
    with Pool(16) as pool:
        res = pool.map(edist, zip(names, refs, preds))

    utt, total_tkn, total_dis = 0, 0, 0
    total_S, total_I, total_D, total_M = 0, 0, 0, 0
    bad_case_stat = {}
    for tmp in res:
        key, ref, hyp, ali, num_tkn, dis, S, I, D, M = tmp
        total_S += S
        total_I += I
        total_D += D
        total_M += M
        total_tkn += num_tkn
        total_dis += dis
        utt += 1
        if log_file is not None and dis != 0:
            err = dis / num_tkn
            print(f'{key}\t{dis}/{num_tkn}={err:.3f}', file=log_file)
            print(' '.join(hyp), file=log_file)
            print(' '.join(ref), file=log_file)
            print('  '.join(ali), file=log_file)
            for index, status in enumerate(ali):
                if status != 'M' and ref[index] != '**':
                    if ref[index] in bad_case_stat:
                        bad_case_stat[ref[index]] += 1
                    else:
                        bad_case_stat[ref[index]] = 1
            print('=' * 50, file=log_file)
    if log_file is not None:
        print('Bad Case:', file=log_file)
        bad_case_stat_sorted = sorted(bad_case_stat.items(),
                                      key=lambda x: (-x[1], x[0]))
        print('\n'.join(f'{word}: {count}'
                        for word, count in bad_case_stat_sorted),
              file=log_file)
    return {
        'utts': utt,
        'ins': total_I,
        'del': total_D,
        'sub': total_S,
        'match': total_M,
        'dis': total_dis,
        'tkn': total_tkn,
    }


def load_finetuned_model(base_model_path, lora_path, device):
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    return model.to(device)


def find_valid_data_dirs(parent_dir, list_file):
    valid_dirs = []
    for root, dirs, files in os.walk(parent_dir, followlinks=True):
        if list_file in files:
            valid_dirs.append(root)
    return valid_dirs


def main():
    args = parse_args()
    # Load and prepare datasets
    data_list = []
    all_data_dirs = []
    for parent_dir in args.data_dirs:
        all_data_dirs.extend(find_valid_data_dirs(parent_dir, args.list_file))
    for data_dir in all_data_dirs:
        with open(osp.join(data_dir, args.list_file),
                  encoding='utf-8') as txt_file:
            lines = txt_file.readlines()

        for line in lines:
            parts = line.strip().split('\t')
            assert len(parts) == 2
            file_name, transcription = parts
            data_list.append({
                'audio':
                osp.join(osp.abspath(data_dir), file_name),
                'transcription':
                transcription,
                'sub_dataset':
                osp.basename(data_dir),
                'name':
                file_name
            })

    dataset = Dataset.from_list(data_list).cast_column('audio',
                                                       datasets.Audio())

    # Create log dir
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(args.log_path) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'>>> Results will be saved to {results_dir}')

    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    ).to(args.device)

    if args.lora_path:
        model = load_finetuned_model(args.model_id, args.lora_path,
                                     args.device)
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        ).to(args.device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    pipe = pipeline('automatic-speech-recognition',
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch.float16,
                    chunk_length_s=args.chunk_length,
                    device=args.device)
    pipe.model.generation_config.forced_decoder_ids = (
        pipe.tokenizer.get_decoder_prompt_ids(language=args.language,
                                              task='transcribe'))

    prompt_ids = processor.get_prompt_ids(args.prompt,
                                          return_tensors='pt')
    prompt_ids = prompt_ids.to(args.device)

    # Generate predictions
    preds = []
    for out in tqdm.tqdm(pipe(KeyDataset(dataset, 'audio'),
                              generate_kwargs={'prompt_ids': prompt_ids},
                              batch_size=args.batch_size),
                         total=len(dataset)):
        pred = out['text'].strip()
        preds.append(pred)

    # Calculate each metric
    table = Table()
    table.add_column('Dataset', style='magenta')
    table.add_column('All utts', justify='right', style='green')
    table.add_column('Ins', justify='right', style='green')
    table.add_column('Del', justify='right', style='green')
    table.add_column('Sub', justify='right', style='green')
    table.add_column('Match', justify='right', style='green')
    table.add_column('WER', justify='right', style='red')

    refs = dataset['transcription']
    names = dataset['name']
    sub_dataset_names = sorted(list(set(dataset['sub_dataset'])))
    for idx, sub_dataset_name in enumerate(sub_dataset_names):
        sub_indices = [
            i for i, name in enumerate(dataset['sub_dataset'])
            if name == sub_dataset_name
        ]
        sub_names = [names[i] for i in sub_indices]
        sub_preds = [preds[i] for i in sub_indices]
        sub_refs = [refs[i] for i in sub_indices]
        with open(results_dir / f'{sub_dataset_name}.log', 'w') as f:
            sub_results = evaluate_pred(sub_names,
                                        sub_refs,
                                        sub_preds,
                                        log_file=f)
        extra_line = idx == len(sub_dataset_names) - 1
        table.add_row(
            sub_dataset_name,
            f"{sub_results['utts']}",
            f"{sub_results['ins']}",
            f"{sub_results['del']}",
            f"{sub_results['sub']}",
            f"{sub_results['match']}",
            f"{sub_results['dis']} / {sub_results['tkn']} = {sub_results['dis']/sub_results['tkn']:.4f}",
            end_section=extra_line)
    overall_results = evaluate_pred(names, refs, preds)
    table.add_row(
        'Overall', f"{overall_results['utts']}", f"{overall_results['ins']}",
        f"{overall_results['del']}", f"{overall_results['sub']}",
        f"{overall_results['match']}",
        f"{overall_results['dis']} / {overall_results['tkn']} = {overall_results['dis']/overall_results['tkn']:.4f}"
    )
    with open(results_dir / 'results.log', 'w') as f:
        rprint(table, file=f)
    rprint(table)


if __name__ == '__main__':
    main()
