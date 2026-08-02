import numpy as np


def map3_metric(eval_pred):
    logits, labels = eval_pred
    top3 = np.argsort(-logits, axis=1)[:, :3]
    scores = []
    for prediction, label in zip(top3, labels):
        if prediction[0] == label:
            scores.append(1.0)
        elif prediction[1] == label:
            scores.append(0.5)
        elif prediction[2] == label:
            scores.append(1 / 3)
        else:
            scores.append(0.0)
    return {
        "MAP@3": np.mean(scores),
        "accuracy": np.mean(np.argmax(logits, axis=1) == labels),
    }
