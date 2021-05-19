import torch


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
train_batches = list(train_iterator)
valid_batches = list(valid_iterator)

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