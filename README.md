# Khmer News Classification Using Traditional Machine Learning Algorithms

A **reproducible experimental workflow** for classifying Khmer-language news articles with
traditional machine learning, built on the [KhNeAC dataset](https://huggingface.co/datasets/sopagnaheang/KhNeAC).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-orange)
![Dataset](https://img.shields.io/badge/dataset-KhNeAC%20(HF)-yellow)

> Course project — *Fundamentals of Machine Learning* · Data Science Program (Year 4)

---

## Overview

Cambodian news platforms publish large volumes of Khmer articles daily, yet manual category
labeling remains slow and inconsistent. This project studies which **traditional** ML algorithm
best classifies Khmer news — and how **Khmer word segmentation** and **feature representation**
affect performance. It is designed as an *experimental study*, not just a predictive program:

- **6 dataset configurations** (raw, clean, normalize, com_seg, mor_seg, kcc_seg) turn segmentation into a controlled experimental factor.
- **4 algorithms** — Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
- **3 feature representations** — Bag of Words, word-level TF-IDF, character-level TF-IDF.
- **Leakage-free protocol** — validation-only tuning, one-shot test evaluation, macro-F1 as the primary metric.
- **Configuration-driven** — fork it and adapt by editing one cell (see the Fork Guide inside the notebook).

## Repository Structure

```text
khmer-news-classification/
├── notebooks/
│   └── Khmer_News_Classification_Workflow.ipynb   # full pipeline (31 cells, 10 sections)
├── docs/
│   ├── project_proposal.pdf                       # academic proposal (submitted to supervisor)
│   └── methodology_guide.pdf                      # design rationale for every notebook decision
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

`artifacts/` (saved models, results.csv) is created at runtime and git-ignored.

## Quickstart

**Option A — Google Colab (zero setup, recommended)**

1. Upload/open `notebooks/Khmer_News_Classification_Workflow.ipynb` in [Colab](https://colab.research.google.com).
2. Runtime → Run all. Runtime ≈ **10–20 min** on the free CPU tier.

**Option B — Local**

```bash
git clone https://github.com/<your-username>/khmer-news-classification.git
cd khmer-news-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/Khmer_News_Classification_Workflow.ipynb
```

> **Smoke test first:** set `CONFIG["sample_size"] = 1000` in the notebook's configuration cell,
> run end-to-end in ~2 minutes, then set it back to `None` for the full study.

## Dataset

[KhNeAC](https://huggingface.co/datasets/sopagnaheang/KhNeAC) — 7,344 labeled Khmer news articles,
loaded automatically from the Hugging Face Hub (no manual download).

| Split | Train | Validation | Test |
|---|---|---|---|
| Articles | 5,140 | 661 | 1,543 |

**Six configurations:** `raw` · `clean` · `normalize` · `com_seg` (compound) · `mor_seg` (morpheme) · `kcc_seg` (Khmer Character Cluster)

**Six classes:** `economic` · `entertainment` · `life` · `politic` · `sport` · `technology`

## Experimental Protocol

| ID | Features | Model | Purpose |
|---|---|---|---|
| E1 | BoW | Naive Bayes | Baseline (H1) |
| E2 | TF-IDF (word) | Naive Bayes | Feature comparison (H1) |
| E3 | TF-IDF (word) | Logistic Regression | Linear comparator |
| E4 | TF-IDF (word) | Linear SVM | Strong sparse-text model (H3) |
| E5 | TF-IDF (word) | Random Forest | Non-linear ensemble |
| E6 | TF-IDF (char) | Linear SVM | Robustness to segmentation (H5) |
| E7 | BoW | Logistic Regression | BoW/linear interaction |
| SEG-* | all six configs | Linear SVM | Segmentation study (H2) |

Hyper-parameters are tuned **on the validation split only** (macro-F1); the test split is
evaluated **exactly once**. Five pre-registered hypotheses (H1–H5) are tested in §8 of the notebook.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Naive Bayes | *to be reported* | *to be reported* |
| Logistic Regression | *to be reported* | *to be reported* |
| Linear SVM | *to be reported* | *to be reported* |
| Random Forest | *to be reported* | *to be reported* |

> Run the notebook, then update this table from `artifacts/results.csv` (and §8 for per-class
> and segmentation findings). No values are asserted before they are measured.

## Reproducibility

- Fixed seed (`SEED = 42`) for all stochastic components; environment versions printed at runtime.
- Official dataset splits are used as-is — no re-splitting, no leakage.
- The winning pipeline (vectorizer + classifier as one object) is saved to `artifacts/` via `joblib`.

## Cite This Repository

GitHub renders **Cite this repository** from [`CITATION.cff`](CITATION.cff). BibTeX:

```bibtex
@software{khmer_news_classification_2026,
  title   = {Khmer News Classification Using Traditional Machine Learning Algorithms},
  author  = {[Author Name]},
  year    = {2026},
  url     = {https://github.com/<your-username>/khmer-news-classification},
  note    = {Course project, Fundamentals of Machine Learning}
}
```

If you use the **dataset**, please cite its authors:
CADT Inclusive Digital Innovation Research Lab (IDRI), *KhNeAC: Khmer News Article Classification
Dataset*, Hugging Face — https://huggingface.co/datasets/sopagnaheang/KhNeAC
(derived from `CADT-IDRI/Khmer_News_classification`).

## License

Code is released under the [MIT License](LICENSE). The KhNeAC dataset is governed by the terms of
its original publishers — see the dataset card on Hugging Face.

## Acknowledgments

- Cambodia Academy of Digital Technology (CADT) — IDRI, for the KhNeAC dataset.
- The Khmer NLP community for segmentation resources (KCC, compound/morpheme variants).
