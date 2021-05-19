from typing_extensions import Required
from model.CBoW import CBoWTextClassifier
from trainer.train import trainer
import fasttext
import argparse
from transformers import ElectraForSequenceClassification, BertForSequenceClassification

def build_model(model_type, vocab_size):
    if model_type == 'cbow':
        vocab_size = 30000
        return CBoWTextClassifier(vocab_size=vocab_size, num_labels=2, embed_dim=32)   
    return pretrained_model(model_type)

def pretrained_model(model_type):
    if model_type == 'electra':
        return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=2)
    else: # mBERT
        return BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # Directory paths
    parser.add_argument('--data-dir', required=True, default='./data')
    parser.add_argument('--output-dir', required=True, default='./output')
    # Model 
    parser.add_argument('--model_type', required=True, choice=['cbow','fasttext','mbert','electra'])
    parser.add_argument('--checkpoint', default='./model/checkpoint')
    # Train/test commands
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    # Hyperparameters
    parser.add_argument('--search_hparam', action="store_true")
    parser.add_argument('--learning-rate', type=int, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    #  1. read data 
    # input : source data / output : dataloader
    #   모델 input 형태로 변형
        # 1. tokenize
        # 2. vocab 생성
        # 3. dataloader 생성
    # 2. 모델 불러오고
    model = build_model(args.model)
    # if model_type == 'fasttext':
    #     model = fasttext.load_model('cc.ko.300.bin')
    #     model = fasttext.train_supervised(input='./train.txt')
    # 3. trainer에서 학습하고 (model, dataloader) 넣어주기
    # 4. test, inference (model, dataloader) 




