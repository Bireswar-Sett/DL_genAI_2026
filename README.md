# DL_genAI_2026

This repository has been refactored into a course-style project structure with reusable model, trainer, dataset, and utility modules.

## Project structure

- `src/models/bert_model.py`: builds a BERT multiple-choice model.
- `src/models/roberta_model.py`: builds a RoBERTa multiple-choice model.
- `src/trainer.py`: custom training loop with evaluation and best-model checkpointing.
- `src/dataset.py`: MCQ dataset tokenization and Torch formatting helpers.
- `src/metrics.py`: MAP@3 evaluation metric.
- `src/utils.py`: device and reproducibility utilities.

The notebook has been updated to use the custom trainer and modular source code instead of `transformers.Trainer`.
