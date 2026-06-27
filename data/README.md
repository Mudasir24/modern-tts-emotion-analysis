# Data

This repository uses the **Emotional Speech Dataset (ESD)** as the source of natural emotional speech and includes synthetic speech generated using **CosyVoice-300M-Instruct** and **Parler-TTS Mini v1**.

## Dataset Sources

### Natural Speech (ESD)

The Emotional Speech Dataset (ESD) contains emotional speech recordings from 20 speakers (10 English and 10 Chinese) across five emotions: **Angry, Happy, Neutral, Sad,** and **Surprise**. We use only the 10 English speakers (IDs 0011–0020) in this study.

* ESD GitHub: https://github.com/HLTSingapore/Emotional-Speech-Data
* ESD Download Page: https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view

### Synthetic Speech

The generated synthetic speech used in this project can be downloaded from the following links:

* **CosyVoice-300M-Instruct:** https://drive.google.com/drive/folders/1NipcxwWmuNKDSUjEle1v6F915iLZInLJ?usp=drive_link
* **Parler-TTS Mini v1:** https://drive.google.com/drive/folders/1Vg6zhV7XF7v-nR2V3XQ3TjD8AZ4FU799?usp=drive_link

## Expected Directory Structure

```text
data/
├── natural/
│   ├── 0011/
│   ├── 0012/
│   ├── ...
│   └── 0020/
│
├── cosyvoice/
│   ├── Angry/
│   ├── Happy/
│   ├── Sad/
│   └── Surprise/
│
└── parlertts/
    ├── Angry/
    ├── Happy/
    ├── Sad/
    └── Surprise/
```

## Notes

* The repository does **not** store large audio files directly due to GitHub storage limitations.
* Please download the datasets from the links above and place them in the appropriate directories.
* Embeddings (`.npy`), model checkpoints, and cached artifacts are intentionally excluded and can be regenerated using the provided scripts.
