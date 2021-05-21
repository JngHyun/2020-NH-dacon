import datetime
import random
import time

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


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

    valid_batches = list(valid_iterator)


def train(model, train_iterator, valid_iterator, epochs):
    train_batches = list(train_iterator)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        t0 = time.time()

        loss_sum = 0
        n_batches = 0

        model.train()

        # ========================================
        #               Training
        # ========================================
        for batch in train_batches:
            # Compute the output scores.
            scores = model(batch.text)
            # Then the loss function.
            loss = criterion(scores, batch.label)

            # Compute the gradient with respect to the loss, and update the parameters of the model.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        train_loss = loss_sum / n_batches
        # history['train_loss'].append(train_loss)

        # After each training epoch, we'll compute the loss and accuracy on the validation set.
        n_correct = 0
        # n_valid = len(valid)
        n_valid = len(valid_iterator)
        loss_sum = 0
        n_batches = 0

        # Calling model.train() will disable the dropout layers.
        model.eval()

        # ========================================
        #               Validation
        # ========================================
        for batch in valid_batches:
            scores = model(batch.text)
            n_corr_batch, loss_batch = evaluate_validation(
                scores, criterion, batch.label
            )
            loss_sum += loss_batch
            n_correct += n_corr_batch
            n_batches += 1
        val_acc = n_correct / n_valid
        val_loss = loss_sum / n_batches
        t1 = time.time()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {i+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val acc: {val_acc:.4f}, time = {t1-t0:.4f}"
            )
            torch.save(
                model.state_dict(), f"./model/checkpoint/Deeping_source_CBoW_{epoch}.pt"
            )

    print("")
    print("Training complete!")


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
