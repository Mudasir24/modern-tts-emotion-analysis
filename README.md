# Analysis of Emotional Speech Synthesis in Modern TTS Systems

<p align="center">
  <b>Embedding-based evaluation of emotional fidelity in modern Text-to-Speech systems.</b>
</p>

---

## Overview

Modern Text-to-Speech (TTS) systems can generate highly natural and expressive speech, yet accurately reproducing emotional expression remains challenging. Conventional evaluation metrics such as Mean Opinion Score (MOS) provide limited insight into how effectively emotions are preserved.

This project presents an **embedding-based framework** for evaluating emotional speech synthesis by comparing:

* **CosyVoice-300M-Instruct**
* **Parler-TTS Mini v1**

against natural speech from the **Emotional Speech Dataset (ESD)**.

Speech representations are extracted using **WavLM Large embeddings** and analyzed using dimensionality reduction, similarity analysis, classification experiments, and subjective listening studies.

---

## Key Findings

* Synthetic speech generally occupies regions of the embedding space similar to natural speech.
* **CosyVoice** consistently produces embeddings that are closer to natural emotional speech than **Parler-TTS**.
* CosyVoice demonstrates substantially better emotion separability than Parler-TTS.
* WavLM embeddings are influenced more strongly by **speaker identity** than **emotional characteristics**.
* Neither system fully reproduces the richness of natural emotional speech.

---

## Methodology

```text
Natural Speech (ESD)
            │
            ├── CosyVoice Synthesis
            ├── Parler-TTS Synthesis
            │
            ▼
      WavLM Large Embeddings
            │
            ▼
      Speaker Centering
            │
            ▼
 PCA • t-SNE • UMAP • LDA
            │
            ▼
 Classification & Similarity
            │
            ▼
      MOS / EMOS Evaluation
```

---

## Repository Structure

```text
modern-tts-emotion-analysis/
│
├── data/
├── cosyvoice/
├── parlertts/
├── wavlm/
├── analysis/
├── results/
└── README.md
```

---

## Dataset

This project uses:

* **Natural Speech:** Emotional Speech Dataset (ESD)
* **Synthetic Speech:** Generated using CosyVoice-300M-Instruct and Parler-TTS Mini v1.

See `data/README.md` for download links and directory structure.

---

## Installation

```bash
git clone https://github.com/Mudasir24/modern-tts-emotion-analysis.git
cd modern-tts-emotion-analysis

pip install -r requirements.txt
```

---

## Generating Synthetic Speech

### CosyVoice

```bash
python cosyvoice/generate_esd.py
```

### Parler-TTS

```bash
python parlertts/generate_esd.py
```

---

## Extracting WavLM Embeddings

```bash
python wavlm/extract_embeddings.py
```

---

## Running Analyses

```bash
python analysis/cosine_similarity.py
python analysis/pca.py
python analysis/tsne.py
python analysis/umap.py
python analysis/lda.py
python analysis/random_forest.py
```

---

## Results

The complete report and presentation can be found in:

```text
report/
├── Project_Report.pdf
└── Presentation.pdf
```

Additional visualizations and sample outputs are available in:

```text
results/
```

---

## Citation

```bibtex
@misc{khanna2026moderntts,
  title={Analysis of Emotional Speech Synthesis in Modern TTS Systems},
  author={Jiya Khanna, Mohammed Mudasir Ahmed and Simran Chhetry},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/Mudasir24/modern-tts-emotion-analysis}
}
```

---

## Acknowledgements

This work was carried out as part of the **IASNLP 2026 Summer School on AI** under the guidance of:

* Namrata Mokshagundam
* Jyothika Goparaju

---

## License

This project is released under the MIT License.
