import numpy as np
import pandas as pd


ANSWER_LABELS = np.array(["A", "B", "C", "D", "E"])


def submission_from_scores(scores, sample_submission_path, output_path="/kaggle/working/submission.csv"):
    top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    prediction_strings = [" ".join(ANSWER_LABELS[idx]) for idx in top3]
    submission = pd.read_csv(sample_submission_path)
    submission["Prediction"] = prediction_strings
    submission.to_csv(output_path, index=False)
    return submission


def generate_submission(test_dataset, trainer, sample_submission_path, output_path="/kaggle/working/submission.csv"):
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    return submission_from_scores(logits, sample_submission_path, output_path)
