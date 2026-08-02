from transformers import AutoModelForMultipleChoice


def get_model():
    return AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
