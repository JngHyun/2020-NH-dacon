# library
import csv
import random

import pandas as pd
import torch
from torchtext.data import BucketIterator, Dataset, Example, Field, LabelField


def read_data(data_dir, command, model_type):
    input_file = data_dir + f"/news_{command}.csv"
    if model_type == "cbow":
        return load_torchtext(input_file, command)
    return pd.read_csv(input_file)


def load_torchtext(input_file, command):
    TEXT = Field(sequential=True, tokenize=lambda x: x.split())
    LABEL = LabelField(is_target=True)
    datafields = [("text", TEXT), ("label", LABEL)]

    with open(input_file, encoding="utf-8") as f:
        raw_data = csv.DictReader(f)
        examples = []
        for row in raw_data:
            content = row.get("content")
            label = row.get("info", -1)
            if label != -1:
                examples.append(Example.fromlist([content, label], datafields))
            else:
                examples.append(Example.fromlist([content], datafields))

    data = Dataset(examples, datafields)

    if command == "train":
        train_data, valid_data = data.split(
            split_ratio=0.9, random_state=random.seed(42)
        )
        TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        vocab_size = len(TEXT.vocab)

        return (train_data, valid_data, vocab_size)

    return data


def build_loader(data_dir, command, model_type, batch_size):
    data = read_data(data_dir, command, model_type)

    # device setting : cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if command == "train":
        if model_type == "cbow":
            train_data = data[0]
            valid_data = data[1]
            vocab_size = data[2]

            train_dataloader, valid_dataloader = BucketIterator.splits(
                (train_data, valid_data),
                batch_size=batch_size,
                sort_within_batch=True,
                sort_key=lambda x: len(x.text),
                device=device,
            )

            return train_dataloader, valid_dataloader, vocab_size
        else:
            raise NotImplementedError

    test_dataloader = BucketIterator(
        data,
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device,
    )
    return test_dataloader
