from datetime import datetime
import torch
from torch import nn


def validate_batch(model, val_loader, loss_fn, device):
    model.train(False)
    running_loss = 0.0
    correct = 0
    for i, val_data in enumerate(val_loader):
        inputs, true_labels = val_data[0].to(device).double(), val_data[1].to(device)
        
        # calculate loss
        output = model(inputs)
        loss = loss_fn(output, true_labels)
        running_loss += loss.item()

        # calculate accuracy
        pred_labels = torch.argmax(output, dim=1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=1, keepdim=False)
        correct += (pred_labels == true_labels).float().sum()

    avg_val_loss = running_loss / (i + 1)
    return avg_val_loss, correct


def train_epochs(data, **train_args):

    model = train_args["model"]
    loss_fn = train_args["loss_fn"]
    optimizer = train_args["optimizer"]
    device = train_args["device"]
    n_epochs = train_args["n_epochs"]
    writer = train_args["writer"]
    save_prefix = train_args["save_prefix"]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = 1e6
    best_val_acc = 0.0

    train_loss = []
    val_loss = []
    val_acc = []

    for epoch in range(n_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_epoch(model, data.train_loader, loss_fn, optimizer, device, epoch, writer)
        
        avg_val_loss, correct = validate_batch(model, data.val_loader, loss_fn, device)

        print('LOSS train {} val {}'.format(avg_train_loss, avg_val_loss))
        print('ACCURACY val {}\n'.format(correct / data.N_val))
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        val_acc.append((correct / data.N_val).cpu())

        writer.add_scalars('training vs. validation loss',
                        { 'training' : avg_train_loss, 'validation': avg_val_loss },
                        epoch + 1)
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'weights/{}_{}_{}'.format(save_prefix, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
        best_val_acc = max(best_val_acc, correct / data.N_val)
        
    return train_loss, val_loss, val_acc


def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch_index, tb_writer):
    running_loss = 0
    last_loss = 0
    total_correct = 0

    for i, data in enumerate(data_loader):
        inputs, true_labels = data[0].to(device).double(), data[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)

        # calculate loss
        loss = loss_fn(output, true_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # calculate accuracy
        pred_labels = torch.argmax(output, dim=1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=1, keepdim=False)
        correct = (pred_labels == true_labels).float().sum()
        total_correct += correct

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            # print('    batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('loss/train', last_loss, tb_x)
            running_loss = 0.
        
    return last_loss
