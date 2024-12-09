
import argparse
import re

import torch
import whisper
from peft import PeftModel
from transformers import WhisperForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id',
                        type=str,
                        default='openai/whisper-large-v3-turbo',
                        help='Model ID')
    parser.add_argument('--openai_model_name',
                        type=str,
                        default='large-v3',
                        help='Openai Model ID')
    parser.add_argument('--lora-path',
                        type=str,
                        required=True,
                        help='Path to Lora file')
    parser.add_argument('--export-to', type=str, required=True)
    args = parser.parse_args()
    return args


def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text


def main():
    args = parse_args()

    print(f"Hugging face id:{args.model_id}")
    # Load HF Model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        # torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()

    hf_state_dict = model.state_dict()
    all_keys = list(hf_state_dict.keys())

    # Rename layers
    openai_state_dict = {}
    for key in all_keys:
        new_key = hf_to_whisper_states(key)
        openai_state_dict[new_key] = hf_state_dict.pop(key)

    # Init Whisper Model and replace model weights
    whisper_model = whisper.load_model(args.openai_model_name, device='cpu')#.half()
    whisper_model.load_state_dict(openai_state_dict, strict=True)

    # Export
    state_dict = {
        'dims': whisper_model.dims.__dict__,
        'model_state_dict': whisper_model.state_dict()
    }
    torch.save(state_dict, args.export_to)
    print(f'Saved to {args.export_to}')


if __name__ == '__main__':
    main()