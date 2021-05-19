#library
import pandas as pd
import torchtext
import torch
import csv

def load_for_cbow_data(path, datafields, c_col=3, l_col=-1):
    with open(path, encoding='utf-8') as f:
      data = csv.reader(f)
      next(data) # skip header
      examples = []
      for line in data:
        doc = line[c_col]
        label = line[l_col]
        examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
    return torchtext.data.Dataset(examples, datafields)


def cbow_databuild(self,path,batch_size,vocab_size):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('the are %d GPU(s) abailable.'%torch.cuda.device_count())
        print('We will use the GPU:',torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')


    TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
    LABEL = torchtext.data.LabelField(is_target=True)
    datafields = [('text', TEXT), ('label', LABEL)]

    data = load_for_cbow_data(path, datafields)
    train, valid = data.split([0.9, 0.1])

    # Build vocabularies from the dataset.
    TEXT.build_vocab(train, max_size=vocab_size)
    LABEL.build_vocab(train)

    train_iterator = torchtext.data.BucketIterator(
        train,
        device=device,
        batch_size=self.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=True)

    valid_iterator = torchtext.data.Iterator(
        valid,
        device=device,
        batch_size=self.batch_size,
        repeat=False,
        train=False,
        sort=False)

    return train_iterator, valid_iterator

def load_data(path):
  train_data = pd.read_csv(path)
  return train_data
