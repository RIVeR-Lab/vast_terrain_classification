import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn


def eval_model(model, data, device, loss_fn):
    model.train(False)
    running_loss = 0
    correct = 0
    all_true_labels = []
    all_pred_labels = []

    for i, test_data in enumerate(data.test_loader):
        inputs, true_labels = test_data[0].to(device), test_data[1].to(device)

        # calculate loss
        output = nn.functional.softmax(model(inputs), dim=1)
        loss = loss_fn(output, true_labels)
        running_loss += loss
        
        # calculate accuracy
        pred_labels = torch.argmax(output, dim=1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=1, keepdim=False)
        correct += (pred_labels == true_labels).float().sum()

        pred_labels = list(pred_labels.cpu().numpy())
        true_labels = list(true_labels.cpu().numpy())
        all_pred_labels += pred_labels
        all_true_labels += true_labels

    avg_test_loss = running_loss / (i + 1)
    accuracy = correct / data.N_test
    print('LOSS test {}'.format(avg_test_loss))
    print('ACCURACY test {}\n'.format(accuracy))

    return avg_test_loss, accuracy, all_true_labels, all_pred_labels


def eval_fused_model(model, data_loader, device, loss_fn, N_data):
    model.train(False)
    running_loss = 0
    correct = 0
    all_true_labels = []
    all_pred_labels = []

    for i, test_data in enumerate(data_loader):
        inputs, true_labels = test_data[0].to(device), test_data[1].to(device)

        with torch.no_grad():
            output = nn.functional.softmax(model(inputs), dim=1)
            loss = loss_fn(output, true_labels)
        running_loss += loss.detach().item()

        # calculate accuracy
        pred_labels = torch.argmax(output, dim=-1, keepdim=False)
        # true_labels = torch.argmax(true_labels, dim=-1, keepdim=False)
        correct += (pred_labels == true_labels).float().sum().cpu()

        pred_labels = list(pred_labels.cpu().numpy())
        true_labels = list(true_labels.cpu().numpy())
        all_pred_labels += pred_labels
        all_true_labels += true_labels
        res = np.array(true_labels) == np.array(pred_labels)
        # if False in res:
        #     print(true_labels)
        #     print(pred_labels)
        #     print(np.where(res == False))
        #     input('Hello world:')
    avg_test_loss = running_loss / (i + 1)
    accuracy = correct / N_data
    print('LOSS test {}'.format(avg_test_loss))
    print('ACCURACY test {}\n'.format(accuracy))

    return avg_test_loss, accuracy, all_true_labels, all_pred_labels