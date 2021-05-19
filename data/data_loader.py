#library
import torchtext
import csv

def load_for_cow_data(path, datafields, c_col=3, l_col=-1):
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
    self.path = path
    self.batch_size = batch_size
    self.vocab_size = vocab_size

    TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
    LABEL = torchtext.data.LabelField(is_target=True)
    datafields = [('text', TEXT), ('label', LABEL)]

    data = load_for_cow_data(path, datafields)
    train, valid = data.split([0.9, 0.1])

    # Build vocabularies from the dataset.
    TEXT.build_vocab(train, max_size=vocab_size)
    LABEL.build_vocab(train)

    train_iterator = torchtext.data.BucketIterator(
        train,
        #device=device,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=True)

    valid_iterator = torchtext.data.Iterator(
        valid,
        #device=device,
        batch_size=128,
        repeat=False,
        train=False,
        sort=False)

    return train_iterator, valid_iterator



#bert
1.토크나이저
DataLoader