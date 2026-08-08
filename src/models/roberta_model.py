from transformers import AutoConfig, AutoModelForMultipleChoice


def get_model(dropout = 0.3):
    config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    return AutoModelForMultipleChoice.from_pretrained("FacebookAI/roberta-base", config=config)
