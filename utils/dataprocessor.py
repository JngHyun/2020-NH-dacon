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


def get_datafields(command):
    TEXT = Field(sequential=True, tokenize=lambda x: x.split())
    if command == "train":
        LABEL = LabelField(is_target=True)
        datafields = [("text", TEXT), ("label", LABEL)]
    else:
        ID = Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("id", ID)]
    return datafields


def get_vocab_size(train_data):
    TEXT = Field(sequential=True, tokenize=lambda x: x.split())
    TEXT.build_vocab(train_data)
    vocab_size = len(TEXT.vocab)
    return vocab_size


def load_torchtext(input_file, command):
    datafields = get_datafields(command)

    with open(input_file, encoding="utf-8") as f:
        raw_data = csv.DictReader(f)
        examples = []
        for row in raw_data:
            field1 = row.get("content")
            field2 = row.get("info") if command == "train" else row.get("id")
            examples.append(Example.fromlist([field1, field2], datafields))

    data = Dataset(examples, datafields)

    if command == "train":
        train_data, valid_data = data.split(
            split_ratio=0.9, random_state=random.seed(42)
        )
        vocab_size = get_vocab_size(train_data)

        return (train_data, valid_data, vocab_size)

    return data


# torch text 아닌 경우에 대한 케이스 작성 필요
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
