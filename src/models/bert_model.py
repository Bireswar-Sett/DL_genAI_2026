from transformers import AutoModelForMultipleChoice

def get_bert_model():
    model = AutoModelForMultipleChoice.from_pretrained(
        "bert-base-uncased"
    )
    return model
