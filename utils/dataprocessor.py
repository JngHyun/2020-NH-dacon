import csv
import random
import pandas as pd
import torch
from torchtext.legacy.data import BucketIterator, Dataset, Example, Field, LabelField
from torch.utils.data import TensorDataset, DataLoader,RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer


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
        train_data.fields['text'].build_vocab(train_data) # max_size
        train_data.fields['label'].build_vocab(train_data)
        vocab_size = len(train_data.fields['text'].vocab)
        
        return (train_data, valid_data, vocab_size)

    return data


# torch text 아닌 경우에 대한 케이스 작성 필요
def token_to_ids(sentences, max_seq_len):
  tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
  sentences = ["[CLS]"+str(sentence)+"[SEP]" for sentence in sentences]
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  MAX_LEN = max_seq_len
  input_ids = pad_sequences(input_ids,maxlen=MAX_LEN, dtype="long",truncating = "post",padding = "post")
  return input_ids

def convert_to_tensordata(inputs, labels, masks):
  t_inputs = torch.tensor(inputs)
  t_labels = torch.tensor(labels)
  t_masks = torch.tensor(masks)

  return TensorDataset(t_inputs, t_masks, t_labels)


def build_loader(data_dir, command, model_type, batch_size):
    data = read_data(data_dir, command, model_type)

    # device setting : cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if command == "train":
        if model_type == "cbow":
            train_data, valid_data, vocab_size = data

            train_dataloader, valid_dataloader = BucketIterator.splits(
                (train_data, valid_data),
                batch_size=batch_size,
                sort_within_batch=True,
                sort_key=lambda x: len(x.text),
                device=device,
            )

            return train_dataloader, valid_dataloader, vocab_size

        if model_type == "electra":
            sentences = data['content']
            labels = data['info'].values

            input_ids = token_to_ids(sentences,max_seq_len=500)
            attention_masks=[]

            for seq in input_ids:
                seq_mask = [float(i>0) for i in seq]
                attention_masks.append(seq_mask)

            train_inputs, validation_inputs,train_labels,validation_labels = train_test_split(input_ids,labels,random_state=2020,test_size=0.1)
            train_masks, validation_masks,_,_=train_test_split(attention_masks,input_ids,random_state=2020,test_size=0.1)


            train_data = convert_to_tensordata(train_inputs, train_labels, train_masks)
            train_dataloader = DataLoader(train_data,
                                         sampler=RandomSampler(train_data),
                                         batch_size=int(batch_size))

            validation_data = convert_to_tensordata(validation_inputs, validation_labels, validation_masks)
            valid_dataloader = DataLoader(validation_data,
                                            sampler=SequentialSampler(validation_data),
                                            batch_size=int(batch_size))
            return train_dataloader, valid_dataloader
        
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
