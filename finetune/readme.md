# Whisper Finetuning via SageMaker Training Job

Please open a SageMaker Notebook instance and follow the steps in `start.ipynb`.
Please modify the `role`, `source_dir`, `data_path` according to your setting.

We also prepare a local infrecne script `train_local.py` for debug usage (you need to use GPU instance like ml.g5.2xlarge).

```bash
python train_local.py --data_dir {your_s3_data_path}
```

## Data Format

Training data is easy to prepare and please following the below format:

```bash
your_dataset/
│── .json
│── audio_001.wav
├── audio_002.wav
├── audio_003.wav
├── audio_004.wav
├── audio_005.wav
├── ...
└── audio_xxx.wav
```

- `your_dataset/`: data path
- `audio_xxx.wav`: WAV audio file
- `successful_texts.json`: json file with GT info like the following format:

```json
{
    "f21e66b2-5af9-4ad6-a84d-8018ad284c67": "What spell should I use in the mid lane?",
    "b119afd8-0128-4a5a-b884-23a610203442": "What the fuck is going on with the map!!",
    "2737efd0-c926-425c-a182-19ca81a581ef": "Is the overpowered character too strong?",
}
```
