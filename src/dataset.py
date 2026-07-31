from datasets import Dataset


def prepare_mcq_dataset(dataframe, tokenizer, max_length=256, include_labels=True):
    dataframe = dataframe.copy()
    if include_labels and "label" in dataframe.columns:
        dataframe = dataframe.rename(columns={"label": "labels"})

    dataset = Dataset.from_pandas(dataframe)

    def tokenize_mcq(example):
        questions = [example["prompt"] for _ in range(5)]
        choices = [example[option] for option in ["A", "B", "C", "D", "E"]]
        tokenized = tokenizer(
            questions,
            choices,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            **({"labels": example["labels"]} if include_labels and "labels" in example else {})
        }

    dataset = dataset.map(tokenize_mcq)
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns("__index_level_0__")

    columns = ["input_ids", "attention_mask"]
    if include_labels and "labels" in dataset.column_names:
        columns.append("labels")

    dataset.set_format(type="torch", columns=columns)
    return dataset
