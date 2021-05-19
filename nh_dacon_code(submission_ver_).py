# -*- coding: utf-8 -*-
"""NH_dacon_code(submission ver.)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OOjcc47HJR_6u4Ps0sQCSoOF9n_X1Cp8

# Environment
Implemented on Colab Notebook
- OS : ubuntu 18.04
- Library : PyTorch-1.7.0+cu101, Transformers-4.1.1

# Test Inference
"""

from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

current_path = Path(os.getcwd())

base_path = current_path / "drive" / "My Drive" /"NH_dacon"
os.chdir(base_path)

# 필요 패키지 설치
!pip install transformers

# 제공된 테스트 파일로부터 dataframe으로 로드
import pandas as pd
test = pd.read_csv('./data/news_test.csv')

# 시간 측정 시작
import time
start = time.time()

# Library 불러오기
import pandas as pd
import numpy as np
import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

# GPU 유무에 따른 device 선언
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('the are %d GPU(s) abailable.'%torch.cuda.device_count())
  print('We will use the GPU:',torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print('No GPU available, using the CPU instead.')

# Tokenizer 로드
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

# model(pretrained koelectra-small-v3 model), optimizer 선언
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator",num_labels=2)
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

# 학습 체크포인트로부터 model, optimizer state 로드
model_state, optimizer_state = torch.load("./data/checkpoint")
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)

# model을 device에 로드
model.to(device)

# model을 evaluation mode로 변경
model.eval()

# 형태소 분석 + 전처리
def convert_input_data(sentences):
    # 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 256

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정 패딩 부분은 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return TensorDataset(inputs, masks)

# 테스트 데이터를 학습하기 위한 데이터로 변환하여 dataloader 선언
test_data = convert_input_data(test['content'])
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=64)

# 결과를 담을 딕셔너리 선언
result_dict = {"id":[],"info":[]}
result_dict["id"].extend(test['id'].values.astype(str))

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
  # 배치를 device에 로드
  batch = tuple(t.to(device) for t in batch)
  
  # 배치에서 데이터 추출
  b_input_ids, b_input_mask = batch
  
  # 그래디언트 계산 안함
  with torch.no_grad():     
      # Forward 수행
      outputs = model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask)
  # logit 구함
  logit = outputs[0]
  logits = logit.detach().cpu().numpy()

  # logit 기반으로 예측 라벨 구함
  predicted_label = np.argmax(logits, axis=1).flatten()
  result_dict["info"].extend(predicted_label)

# test 소요시간 출력
print(time.time() - start)

# 결과값을 담은 딕셔너리로부터 dataframe을 생성하고, 제출용 csv파일로 출력하여 저장
sub = pd.DataFrame(result_dict)
sub.to_csv('./data/submission_256.csv', index=False, header=True)

