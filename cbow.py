# -*- coding: utf-8 -*-
"""CBoW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fI5TXXTM_4NQ968fhqZNvvJaOL2IpedL

Reference : http://www.cse.chalmers.se/~richajo/nlp2019/l2/Text%20classification%20using%20a%20CBoW%20representation.html
"""

from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

current_path = Path(os.getcwd())
base_path = current_path / "drive" / "My Drive" /"NH_dacon"
os.chdir(base_path)

# Commented out IPython magic to ensure Python compatibility.
import torch
from torch import nn

import torchtext

from collections import defaultdict
import time

import pandas as pd
import csv
import numpy as np

import matplotlib.pyplot as plt

# Just to make the plots look nice.
# %config InlineBackend.figure_format = 'retina' 
plt.style.use('seaborn')

class CBoWTextClassifier(nn.Module):
  def __init__(self, text_field, class_field, emb_dim, n_hidden=20, dropout=0.1):
    super().__init__()
    voc_size = len(text_field.vocab)
    n_classes = len(class_field.vocab)
    self.embedding = nn.Embedding(voc_size, emb_dim)
    self.hidden_layer = nn.Linear(emb_dim, n_hidden)
    self.top_layer = nn.Linear(n_hidden, n_classes) # A linear output layer
    self.dropout = nn.Dropout(dropout)

  def forward(self, docs):
    # The words in the documents are encoded as integers. The shape of the documents
    # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
    # and max_len is the maximal length of a document in the batch.

    # First look up the embeddings for all the words in the documents.
    # The shape is now (max_len, n_docs, emb_dim)
    embedded = self.embedding(docs)
    # Compute the mean of word embeddings over the documents.
    # The shape is now (n_docs, emb_dim)
    cbow = embedded.mean(dim=0)
    # Apply the dropout layer. (This is only used during training, not during testing.)
    cbow_drop = self.dropout(cbow)
    hidden = torch.relu(self.hidden_layer(cbow_drop))
    scores = self.top_layer(hidden)
    return scores

# # train = pd.read_csv('./news_train.csv')
# data = pd.read_csv('./news_train.csv')
# train_data, valid_data = train_test_split(data,random_state=2020,test_size=0.2)

# train_data[:5]
# valid_data[:5]

# TEST CODE : IF READ AS DATAFRAME
# TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
# LABEL = torchtext.data.LabelField(is_target=True)
# datafields = [('text', TEXT), ('label', LABEL)] 

# def read_data(data, datafields):
#     examples = []
#     for row in data.itertuples():
#         columns = line.strip().split(maxsplit=doc_start)
#         doc = row.content
#         label = row.info
#         examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
#     return torchtext.data.Dataset(examples, datafields)

# train = read_data(train_data, datafields)

import csv
def read_data(data_path, datafields, c_col=3, l_col=-1):
    with open(data_path, encoding='utf-8') as f:
      data = csv.reader(f)
      next(data) # skip header
      examples = []
      for line in data:
        doc = line[c_col]
        # print(doc)
        label = line[l_col]
        # print(label)
        examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
    return torchtext.data.Dataset(examples, datafields)

# TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
# LABEL = torchtext.data.LabelField(is_target=True)
# datafields = [('text', TEXT), ('label', LABEL)]

# # data = pd.read_csv('./news_train.csv')
# # train_data, valid_data = train_test_split(data,random_state=2020,test_size=0.2)
# data = read_data('./news_train.csv', datafields)

def evaluate_validation(scores, loss_function, labels):
    preds = scores.argmax(dim=1)
    n_correct = (preds == labels).sum().item()
    return n_correct, loss_function(scores, labels).item()

# def main():

# We first declare the fields of the dataset: one field for the text, and one for the output label.
# For the text field, we also provide a tokenizer.
# In this case, we can use a simple tokenizer since the text is already tokenized in the file.
TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
LABEL = torchtext.data.LabelField(is_target=True)
datafields = [('text', TEXT), ('label', LABEL)]

# data = pd.read_csv('./news_train.csv')
# train_data, valid_data = train_test_split(data,random_state=2020,test_size=0.2)
data = read_data('./news_train.csv', datafields)
train, valid = data.split([0.9, 0.1])

# Build vocabularies from the dataset.
TEXT.build_vocab(train, max_size=30000)
LABEL.build_vocab(train)

# Declare the model. We'll use the shallow CBoW classifier or the one that has one hidden layer.
model = CBoWTextClassifier(TEXT, LABEL, emb_dim=32)   

print(model)

# Put the model on the device.
device = 'cuda'
model.to(device)

# The BucketIterator groups sentences of similar lengths into "buckets", which reduces the need
# for padding when we create minibatches.
# See here: https://pytorch.org/text/data.html#torchtext.data.BucketIterator
train_iterator = torchtext.data.BucketIterator(
    train,
    device=device,
    batch_size=128,
    sort_key=lambda x: len(x.text),
    repeat=False,
    train=True)

valid_iterator = torchtext.data.Iterator(
    valid,
    device=device,
    batch_size=128,
    repeat=False,
    train=False,
    sort=False)

# Cross-entropy loss as usual, since we have a classification problem.
loss_function = torch.nn.CrossEntropyLoss()

# Adam optimizer. We can try to tune the learning rate to get a fast convergence while avoiding instability.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# To speed up training, we'll put all the batches onto the GPU. This will avoid repeating
# the preprocessing of the batches as well as the GPU communication overhead.
# We can do this because the dataset is not that big so that it fits in the GPU memory.
# Torchtext will handle all administration: mapping text to integers and putting everything into tensors.
train_batches = list(train_iterator)
valid_batches = list(valid_iterator)

# We'll keep track of some indicators and plot them in the end.
history = defaultdict(list)

for i in range(200):
    
    t0 = time.time()
    
    loss_sum = 0
    n_batches = 0

    # Calling model.train() will enable the dropout layers.
    model.train()
    
    # We iterate through the batches created by torchtext.
    # For each batch, we can access the text part and the output label part separately.
    for batch in train_batches:
        
        # Compute the output scores.
        scores = model(batch.text)
        # Then the loss function.
        loss = loss_function(scores, batch.label)

        # Compute the gradient with respect to the loss, and update the parameters of the model.
        optimizer.zero_grad()            
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        n_batches += 1
    
    train_loss = loss_sum / n_batches
    history['train_loss'].append(train_loss)
    
    # After each training epoch, we'll compute the loss and accuracy on the validation set.
    n_correct = 0
    n_valid = len(valid)
    loss_sum = 0
    n_batches = 0

    # Calling model.train() will disable the dropout layers.
    model.eval()

    for batch in valid_batches:
        scores = model(batch.text)
        n_corr_batch, loss_batch = evaluate_validation(scores, loss_function, batch.label)
        loss_sum += loss_batch
        n_correct += n_corr_batch
        n_batches += 1
    val_acc = n_correct / n_valid
    val_loss = loss_sum / n_batches

    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)        
    
    t1 = time.time()

    if (i+1) % 10 == 0:
        print(f'Epoch {i+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val acc: {val_acc:.4f}, time = {t1-t0:.4f}')
    
    output_file = "./NH_CBoW_{}.pt".format(epoch_i + 1)
    torch.save(model.state_dict(), output_file)
    
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.plot(history['val_acc'])
plt.legend(['training loss', 'validation loss', 'validation accuracy'])



import time
start = time.time()

# model.load_state_dict(torch.load("./NH_KoELECTRA_small_v3_3.pt"))
# model.cuda()



import csv
info = result_dict["info"]
id = result_dict["id"]
with open('./submission_koelectra_small_v3_epoch3.csv','w') as f:
    # fieldnames = result_dict.keys()
    w = csv.writer(f,delimiter=',')
    w.writerow(result_dict.keys())
    for v in zip(id,info):
      w.writerow(v)