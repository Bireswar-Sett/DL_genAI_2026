from transformers import AutoModelForMultipleChoice

def get_roberta_model():
    model = AutoModelForMultipleChoice.from_pretrained(
        "roberta-base"
    )
    return model
