# DL_genAI_2026 — MCQ Answer Ranking

A deep learning project for ranking multiple-choice answers. Given a question and 5 answer options (A–E), the model scores each option and predicts the correct one. Built for the [Smart MCQ Solver Challenge](https://www.kaggle.com/competitions/smart-mcq-solver-challenge) on Kaggle.

## What's in here

Three models, all trained and evaluated on the same task:

- **A custom bi-encoder**, built from scratch
- **BERT** (`bert-base-uncased`), fine-tuned
- **RoBERTa** (`roberta-base`), fine-tuned

## Architecture

**Custom bi-encoder (`MCQBiEncoder`)**:
- A shared `TransformerEncoder` (4 layers, `d_model=256`, 8 attention heads) encodes the question and each of the 5 options independently, with no pretrained weights
- Sinusoidal positional encoding
- Mask-aware mean pooling over token embeddings
- Score = cosine similarity between the question embedding and each option embedding, scaled by a learned temperature
- Trained end-to-end with cross-entropy over the 5 similarity scores

**Baselines**: `bert-base-uncased` and `roberta-base`, fine-tuned via `AutoModelForMultipleChoice`, with the question and each option formatted as a 5-way multiple-choice input.

## Repo Structure

```
src/
├── models/
│   ├── custom_model.py   # MCQBiEncoder + TransformerEncoder + PositionalEncoding
│   ├── bert_model.py      # bert-base-uncased multiple-choice wrapper
│   └── roberta_model.py   # roberta-base multiple-choice wrapper
├── trainer.py              # MCQTrainer (custom loop) + HF Trainer setup for baselines
├── metrics.py               # MAP@3 / accuracy / F1 computation
DL-24F2001551-notebook-t22026.ipynb   # full EDA, training, and evaluation notebook
```

## Data

- ~9k question/option rows, 5 classes (A–E), moderate class imbalance (295–442 samples per class)
- 183 exact duplicate rows removed
- Rows sharing an identical prompt but different options were kept as augmentation — the train/validation split uses `GroupShuffleSplit` grouped by prompt text, so no question leaks across the split
- Max token length set to 256, covering ~99% of question+option pairs

## Running

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn wandb
```

The notebook auto-detects Kaggle vs local (`IS_KAGGLE` check) and points to `./data/` locally — drop `train.csv` / `test.csv` from the competition there. A `WANDB_API_KEY` (via Kaggle secrets or local env) enables experiment tracking; training still runs without it.

## Results

Scored via MAP@3 on the Kaggle leaderboard (unseen test set):

| | MAP@3 |
|---|---|
| **Custom bi-encoder** | **0.754** |
| Top leaderboard submission | 0.77 |

The from-scratch bi-encoder landed within ~2 points of the top submission on the competition leaderboard.
