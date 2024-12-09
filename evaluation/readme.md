# Speech Recognition and Evaluation Tool

This tool performs speech recognition on audio files and evaluates the transcription quality using various metrics.

## Features

- Supports multiple input directories
- Batch processing of audio files
- Uses Whisper model for speech recognition
- Supports LoRA fine-tuning
- Calculates Word Error Rate (WER) and other metrics
- Generates detailed logs for each dataset


## Usage

```bash
python speech_recognition_evaluation.py --data-dirs <dir1> <dir2> ... --language <lang> [options]
```

### Arguments

- `--data-dirs`: List of data directories (required)
- `--language`: Language of the audio (required: 'en', 'ms', 'id', or 'zh')
- `--model-id`: Hugging Face model ID (default: 'openai/whisper-large-v3')
- `--batch-size`: Batch size for inference (default: 16)
- `--chunk-length`: Input length for each chunk in seconds (default: 60)
- `--log-path`: Path to the log directory (default: './logs')
- `--device`: Device to use (default: 'cuda')
- `--lora-path`: Path to LoRA file (optional)
- `--prompt`: Pre-defined prompt for the model (optional)

## Example

```bash
python speech_recognition_evaluation.py --data-dirs /path/to/dataset1 /path/to/dataset2 --language en --model-id openai/whisper-large-v3 --batch-size 32 --lora-path /path/to/lora/weights
```

## Data Format

The tool expects a `trans.txt` file in each data directory with the following format:

```
audio_file1.wav    transcription of audio file 1
audio_file2.wav    transcription of audio file 2
...
```

Example:
```
rec1.wav    hello world
rec2.wav    speech recognition is amazing
```

## Output

The tool generates:
1. A summary table with metrics for each dataset and overall results
2. Detailed log files for each dataset, including error analysis

Example output table:

```
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dataset  ┃ All utts┃ Ins ┃ Del ┃ Sub ┃ Match ┃ WER                           ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ dataset1 │     100 │  10 │  15 │  20 │   955 │ 45 / 1000 = 0.0450            │
│ dataset2 │     200 │  25 │  30 │  35 │  1910 │ 90 / 2000 = 0.0450            │
│ Overall  │     300 │  35 │  45 │  55 │  2865 │ 135 / 3000 = 0.0450           │
└──────────┴─────────┴─────┴─────┴─────┴───────┴───────────────────────────────┘
```

