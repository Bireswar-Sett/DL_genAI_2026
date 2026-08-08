from transformers import AutoConfig, AutoModelForMultipleChoice

def get_model(dropout):
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    return AutoModelForMultipleChoice.from_pretrained("bert-base-uncased", config=config)
