# from typing_extensions import Required
from model.CBoW import CBoWTextClassifier
from trainer.trainer import run
import argparse
from utils.dataprocessor import build_loader
from transformers import ElectraForSequenceClassification, BertForSequenceClassification

def build_model(model_type, vocab_size):
    if model_type == 'cbow':
        return CBoWTextClassifier(vocab_size=vocab_size, num_label=2, emb_dim=32)   
    return pretrained_model(model_type)

def pretrained_model(model_type):
    if model_type == 'electra':
        return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=2)
    else: # mBERT
        return BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2)

def set_hyperparameter_dict():
    """ Set your best hyperparameters for your model
    """
    param_dict = {
        'max_seq_len': 384, 
        'train_batch_size': 128,
        'test_batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 2e-5,
    }
    return param_dict

def main(args):
    param_dict = set_hyperparameter_dict()
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    #  1. read data 
    # input : source data / output : dataloader

    if args.do_train:
        train_dataloader, val_dataloader, vocab_size = build_loader(args.data_dir, "train", args.model_type, param_dict['train_batch_size'])
    if args.do_test:
        test_dataloader, vocab_size = build_loader(args.data_dir, "test", args.model_type, param_dict['test_batch_size'])

    if args.model_type == 'cbow':
        if args.do_train:
            train_dataloader, val_dataloader, vocab_size = build_loader(args.data_dir, "train", args.model_type, param_dict['train_batch_size'])
        if args.do_test:
            test_dataloader = build_loader(args.data_dir, "test", args.model_type, param_dict['test_batch_size'])


        #   모델 input 형태로 변형
            # 1. tokenize
            # 2. vocab 생성
            # 3. dataloader 생성
        # 2. 모델 불러오고
        model = build_model(args.model_type, vocab_size=vocab_size)

    if args.model_type == 'electra':
        if args.do_train:
            train_dataloader, val_dataloader = build_loader(args.data_dir, "train", args.model_type, param_dict['train_batch_size'])
        if args.do_test:
            test_dataloader = build_loader(args.data_dir, "test", args.model_type, param_dict['test_batch_size'])
        model = pretrained_model(args.model_type)

    print('success!')
    # checkpoint load
    # if args.checkpoint:
    #     model_state, optimizer_state = torch.load(args.checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
    # if model_type == 'fasttext':
    #     model = fasttext.load_model('cc.ko.300.bin')

    #     model = fasttext.train_supervised(input='./train.txt')
    # 3. trainer에서 학습하고 (model, dataloader) 넣어주기
    run(train_dataloader, val_dataloader, test_dataloader, model, args.checkpoint, args.output_dir, param_dict['num_epochs'], param_dict['learning_rate'])
    # 4. test, inference (model, dataloader) 

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # Directory paths
    parser.add_argument('--data-dir', required=True, default='./data')
    parser.add_argument('--output-dir', required=True, default='./output')
    # Model 
    parser.add_argument('--model-type', required=True, choices=['cbow','fasttext','mbert','electra'])
    parser.add_argument('--checkpoint', type=str)
    # Train/test commands
    parser.add_argument('--do-train', action="store_true")
    parser.add_argument('--do-test', action="store_true")
    # Hyperparameters
    parser.add_argument('--search-hparam', action="store_true")
    args = parser.parse_args()


    if args.do_train == False & args.do_test == False:
        print('Nothing to do !')
        raise NotImplementedError

    main(args)