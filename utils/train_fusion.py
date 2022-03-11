from datetime import datetime
import torch
from torch import nn
import tqdm


def validate_fused_batch(fused_model, val_loader, loss_fn, device):
    fused_model.train(False)
    running_loss = 0.0
    fused_correct = 0
    img_correct = 0
    imu_correct = 0
    spec_correct = 0

    for i, val_data in enumerate(val_loader):
        inputs, true_labels = val_data[0].to(device), val_data[1].to(device)

        img_pred_labels = torch.argmax(inputs[:, :11], dim=-1, keepdim=False)
        imu_pred_labels = torch.argmax(inputs[:, 11:22], dim=-1, keepdim=False)
        spec_pred_labels = torch.argmax(inputs[:, 22:], dim=-1, keepdim=False)

        output = fused_model(inputs)
        loss = loss_fn(output, true_labels)
        running_loss += loss.item()

        # calculate accuracy
        pred_labels = torch.argmax(output, dim=-1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=-1, keepdim=False)

        fused_correct += (pred_labels == true_labels).float().sum().cpu()
        img_correct += (img_pred_labels == true_labels).float().sum().cpu()
        imu_correct += (imu_pred_labels == true_labels).float().sum().cpu()
        spec_correct += (spec_pred_labels == true_labels).float().sum().cpu()
    
    avg_val_loss = running_loss / (i + 1)
    correct = [fused_correct, img_correct, imu_correct, spec_correct]
    return avg_val_loss, correct


def train_fused_epoch(fused_model, train_loader, **train_args):
    loss_fn = train_args["loss_fn"]
    optimizer = train_args["optimizer"]
    device = train_args["device"]
    epoch_index = train_args["epoch_index"]
    tb_writer = train_args["tb_writer"]

    # img_model, imu_model, spec_model = other_models

    running_loss = 0
    last_loss = 0
    correct = 0

    for i, data in tqdm.tqdm(enumerate(train_loader)):
        inputs, true_labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = fused_model(inputs)

        # calculate loss
        loss = loss_fn(output, true_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # calculate accuracy
        pred_labels = torch.argmax(output, dim=-1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=-1, keepdim=False)
        correct += (pred_labels == true_labels).float().sum()

        if (i+1) % 1000 == 0:
            last_loss = running_loss / 1000
            print('    batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('loss/train', last_loss, tb_x)
            # model_path = 'weights/fused_{}_{}.wts'.format(epoch_index + 1, i)
            # torch.save(fused_model.state_dict(), model_path)
            running_loss = 0.
        
    return last_loss

def train_fused_epochs(fused_model, train_loader, val_loader, **train_args):

    loss_fn = train_args["loss_fn"]
    optimizer = train_args["optimizer"]
    device = train_args["device"]
    n_epochs = train_args["n_epochs"]
    writer = train_args["tb_writer"]
    save_prefix = train_args["save_prefix"]
    N_train = train_args["N_train"]
    N_val = train_args["N_val"]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = 1e6
    best_val_acc = 0.0

    train_loss = []
    val_loss = []
    val_acc = []

    for epoch in range(n_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        fused_model.train(True)
        train_epoch_args = {
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "device": device,
            "epoch_index": epoch,
            "tb_writer": writer
        }
        avg_train_loss = train_fused_epoch(
            fused_model, train_loader, **train_epoch_args
        )
        
        avg_val_loss, correct = validate_fused_batch(
            fused_model, val_loader, loss_fn, device
        )
        fused_correct, img_correct, imu_correct, spec_correct = correct

        print('\nLOSS train {} val {}'.format(avg_train_loss, avg_val_loss))
        print('ACCURACY val {}'.format(fused_correct / N_val))
        print('ACCURACY img {}'.format(img_correct / N_val))
        print('ACCURACY imu {}'.format(imu_correct / N_val))
        print('ACCURACY spec {}\n'.format(spec_correct / N_val))
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        val_acc.append(fused_correct / N_val)

        writer.add_scalars('training vs. validation loss',
                        { 'training' : avg_train_loss, 'validation': avg_val_loss },
                        epoch + 1)
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'weights/val_{}_{}_{}.wts'.format(save_prefix, timestamp, epoch)
            torch.save(fused_model.state_dict(), model_path)
        best_val_acc = max(best_val_acc, fused_correct / (len(val_loader) * val_loader.batch_size))
        
    return train_loss, val_loss, val_acc

