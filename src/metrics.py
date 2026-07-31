import torch

def map3_metric(logits, labels):
    top3 = torch.topk(
        logits,
        k=3,
        dim=1
    ).indices

    score = 0.0
    for pred, label in zip(top3, labels):
        pred = pred.tolist()
        label = label.item()
        if label in pred:
            rank = pred.index(label)
            score += 1.0 / (rank + 1)

    return score / len(labels)
