# -*- coding: utf-8 -*-
"""FastText.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pMYxvl-kUqatBGdsToXfd_0jlTK6Nrl5

Reference
"""

from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

current_path = Path(os.getcwd())
base_path = current_path / "drive" / "My Drive" / "NH_dacon" 
os.chdir(base_path)

# %cd '/content/drive/My Drive/NH_dacon' 
!pwd

!pwd

prefix = '__label__'

import csv

# train data
with open('./news_train.csv', encoding='utf-8') as f:
  data = csv.reader(f)
  next(data) # skip header
  examples = []
  for line in data:
    label = prefix + line[-1]
    content = line[3]
    examples.append([label,content])

with open('./train.txt', 'w', encoding='utf-8') as nf:
    for ex in examples:
      nf.write(ex[0]+' '+ex[1]+'\n')

f.close()
nf.close()

examples[:3]

!head -n 3 './train.txt'

import csv

# test data
with open('./news_test.csv', encoding='utf-8') as f:
  data = csv.reader(f)
  next(data) # skip header
  examples = []
  t_ids = []
  for line in data:
    t_ids.append(line[-1])
    examples.append(line[3])

with open('./test.txt', 'w', encoding='utf-8') as nf:
    for ex in examples:
      nf.write(ex+'\n')

f.close()
nf.close()

!head -n 3 './test.txt'

!pip install gensim

!pip install fasttext

import fasttext
model = fasttext.load_model('cc.ko.300.bin')

# preds = model.predict('test.txt')

model = fasttext.train_supervised(input='./train.txt') 
# model.save_model('./model_ft.bin')
# result = classifier.test('test.txt')
# preds = classifier.predict('./test.txt')
# labels = [ p.replace('__label__','') for p in preds ]

# Commented out IPython magic to ensure Python compatibility.
!wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
!unzip v0.9.2.zip
# %cd fastText-0.9.2
!make

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/NH_dacon/fastText-0.9.2
!make
# !pwd

!./fasttext supervised -input ../train.txt -output model

!./fasttext predict-prob model.bin ../train.txt > ../fasttext_train_prob.txt

!./fasttext predict model.bin ../test.txt > ../result.txt

!head -n 3 '../result.txt'

# preds = []
with open('./result.txt') as f:
  preds = f.readlines()

labels = [ p.replace('__label__','') for p in preds ]
labels = [ l.replace('\n','') for l in labels ]

import csv
with open('./submission_fasttext.csv','w') as f:
    # fieldnames = result_dict.keys()
    w = csv.writer(f,delimiter=',')
    w.writerow(['id','info'])
    for v in zip(t_ids,labels):
      w.writerow(v)