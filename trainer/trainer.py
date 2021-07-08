import datetime
import random
import time
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


def evaluate_validation(scores, criterion, labels):
    preds = scores.argmax(dim=1)
    n_correct = (preds == labels).sum().item()
    return n_correct, criterion(scores, labels).item()

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def run(train_dataloader, val_dataloader, test_dataloader, model, model_type, checkpoint, output_dir, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    if train_dataloader and val_dataloader:
        for epoch in range(epochs):
            print(f'Epoch: {epoch+1:02}')
            start_time = time.time()

            train_loss, train_acc = train(model,model_type, train_dataloader, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model,model_type, val_dataloader, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'\nEpoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint/Deeping_source_CBoW_{epoch}.pt"))

        print("Training complete!")

    if test_dataloader is not None:
        result_dict = inference(model, test_dataloader)
        
        # 결과값을 담은 딕셔너리로부터 dataframe을 생성하고, 제출용 csv파일로 출력하여 저장
        output = pd.DataFrame(result_dict)
        output.to_csv(os.path.join(output_dir, 'submission.csv', index=False, header=True))
    
def train(model, model_type, dataloader, optimizer, criterion):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('the are %d GPU(s) abailable.'%torch.cuda.device_count())
        print('We will use the GPU:',torch.cuda.get_device_name(0))
    
    #scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

    total_epoch_loss = 0
    total_epoch_acc = 0
    
    model.train()

    # ========================================
    #               Training
    # ========================================
    for batch in tqdm(dataloader, desc="train"):
        if model_type=='cbow':
            optimizer.zero_grad() 
            text, labels = batch.text, batch.label
            # Compute the output scores.
            preds = model(text)

            # Then the loss function.
            loss = criterion(preds, labels)
            # Compute the gradient with respect to the loss, and update the parameters of the model.
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()

            acc = flat_accuracy(preds.detach().numpy(), labels)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
        
        if model_type == 'electra':
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            acc = flat_accuracy(logits, label_ids)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            #scheduler.step()
            model.zero_grad()

    epoch_loss = total_epoch_loss / len(dataloader)
    epoch_acc = total_epoch_acc / len(dataloader)

    return epoch_loss, epoch_acc
    

def evaluate(model, model_type, dataloader, criterion):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('the are %d GPU(s) abailable.'%torch.cuda.device_count())
        print('We will use the GPU:',torch.cuda.get_device_name(0))
    
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    model.eval()

    # ========================================
    #               Validation
    # ========================================
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluate"):
            if model_type =='cbow':
                text, labels = batch.text, batch.label
                # Compute the output scores.
                preds = model(text)
                
                # Then the loss function.
                loss = criterion(preds, batch.label)
                # Compute the gradient with respect to the loss, and update the parameters of the model.
                acc = flat_accuracy(preds.detach().numpy(), labels)

                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()
            if model_type =='electra':
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch
   
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                acc = flat_accuracy(logits, label_ids)

                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

    epoch_loss = total_epoch_loss / len(dataloader)
    epoch_acc = total_epoch_acc / len(dataloader)

    return epoch_loss, epoch_acc

def inference(model, dataloader):
    result_dict = {"id":[],"info":[]}

    model.eval()

    # ========================================
    #               Validation
    # ========================================
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluate"):
            text, ids = batch.text, batch.ids
            # Compute the output scores.
            preds = model(text)

    # logit 구함
    logit = preds[0]
    logits = logit.detach().cpu().numpy()

    # logit 기반으로 예측 라벨 구함
    preds = np.argmax(logits, axis=1).flatten()
    result_dict["info"].extend(preds)
    result_dict["id"].extend(ids)

    return result_dict

# def pretrained_model_trainer(model, train_dataloader, validation_dataloader, epochs):

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("the are %d GPU(s) abailable." % torch.cuda.device_count())
#         print("We will use the GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("No GPU available, using the CPU instead.")

#     optimizer = AdamW(model.parameters(), lr=3e-4, eps=1e-8)

#     total_steps = len(train_dataloader) * epochs
#     print("total steps : ", total_steps)

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=0, num_training_steps=total_steps
#     )

#     model.zero_grad()

#     for epoch in range(epochs):

#         # ========================================
#         #               Training
#         # ========================================

#         print("")
#         print("======== Epoch {:} / {:} ========".format(epoch + 1, epochs))
#         print("Training...")

#         t0 = time.time()
#         total_loss = 0
#         model.train()

#         for step, batch in enumerate(train_dataloader):
#             if step % 500 == 0 and not step == 0:
#                 elapsed = format_time(time.time() - t0)
#                 print(
#                     "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
#                         step, len(train_dataloader), elapsed
#                     )
#                 )

#             batch = tuple(t.to(device) for t in batch)

#             b_input_ids, b_input_mask, b_labels = batch

#             outputs = model(
#                 b_input_ids,
#                 token_type_ids=None,
#                 attention_mask=b_input_mask,
#                 labels=b_labels,
#             )

#             loss = outputs[0]
#             total_loss += loss.item()
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             optimizer.step()
#             scheduler.step()
#             model.zero_grad()

#         train_loss = total_loss / len(train_dataloader)

#         print("")
#         print("  Average training loss: {0:.2f}".format(train_loss))
#         print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

#         # ========================================
#         #               Validation
#         # ========================================

#         print("")
#         print("Running Validation...")

#         t0 = time.time()

#         model.eval()

#         n_batch = 0
#         eval_loss, eval_accuracy = 0, 0
#         nb_eval_steps, nb_eval_examples = 0, 0

#         for batch in validation_dataloader:
#             batch = tuple(t.to(device) for t in batch)

#             b_input_ids, b_input_mask, b_labels = batch

#             with torch.no_grad():
#                 outputs = model(
#                     b_input_ids, token_type_ids=None, attention_mask=b_input_mask
#                 )

#             logit = outputs[0]
#             logits = logit.detach().cpu().numpy()

#             label_ids = b_labels.to("cpu").numpy()

#             tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#             eval_accuracy += tmp_eval_accuracy
#             nb_eval_steps += 1

#         # print("  Average validation loss: {0:.2f}".format(val_loss))
#         print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
#         print("  Validation took: {:}".format(format_time(time.time() - t0)))
#         torch.save(
#             model.state_dict(),
#             f"./model/checkpoint/Deeping_source_pretrained_{epoch}.pt",
#         )

#     print("")
#     print("Training complete!")

# # 결과를 담을 딕셔너리 선언
# result_dict = {"id":[],"info":[]}
# result_dict["id"].extend(test['id'].values.astype(str))

# # 데이터로더에서 배치만큼 반복하여 가져옴
# for step, batch in enumerate(test_dataloader):
#   # 배치를 device에 로드
#   batch = tuple(t.to(device) for t in batch)

#   # 배치에서 데이터 추출
#   b_input_ids, b_input_mask = batch

#   # 그래디언트 계산 안함
#   with torch.no_grad():
#       # Forward 수행
#       outputs = model(b_input_ids,
#                       token_type_ids=None,
#                       attention_mask=b_input_mask)
#   # logit 구함
#   logit = outputs[0]
#   logits = logit.detach().cpu().numpy()

#   # logit 기반으로 예측 라벨 구함
#   predicted_label = np.argmax(logits, axis=1).flatten()
#   result_dict["info"].extend(predicted_label)

# # 결과값을 담은 딕셔너리로부터 dataframe을 생성하고, 제출용 csv파일로 출력하여 저장
# sub = pd.DataFrame(result_dict)
# sub.to_csv('./data/submission_256.csv', index=False, header=True)