from model.CBoW import CBoWTextClassifier
from trainer.train import trainer
import fasttext
import argparse
from transformers import ElectraForSequenceClassification, BertForSequenceClassification

def pretrained_model(model_type):
    if model_type == 'fasttext':
        model = fasttext.load_model('cc.ko.300.bin')
        model = fasttext.train_supervised(input='./train.txt')
    elif model_type == 'electra':
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=2)
    else: # BERT
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=2)
    return model

if __name__ =='__main__':
    parser = argparse.ArgumentParse()
    parser.add_argument('train')
    parser.add_argument('test')
    parser.add_argument('--model', choice=['cbow','fasttext','kobert','electra'],required=True)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--test-batch-size',type=int)
    parser.add_argument('--epochs',typ=int,default=3)
    args = parser.parse_args()

    
    if args.model == 'cbow':
        text, label = load_data()
        model = CBoWTextClassifier(vocab_size=len(text.vocab), label_num=len(label.vocab), embed_dim=32)   
    else:
        model = pretrained_model(args.model)