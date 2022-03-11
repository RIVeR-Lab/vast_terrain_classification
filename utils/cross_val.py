from models import MLP
from .eval import eval_model
from .train import train_epochs

from datetime import datetime
from itertools import product
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter


def cross_val_mlp_depth_sizes(data, **cross_val_args):

    num_hidden_layers = cross_val_args["num_hidden_layers"]
    hidden_layer_sizes = cross_val_args["hidden_layer_sizes"]
    D = cross_val_args["data_dim"]
    K = cross_val_args["n_classes"]
    device = cross_val_args["device"]
    use_dropout = cross_val_args["use_dropout"]
    lr = cross_val_args["learning_rate"]
    momentum = cross_val_args["momentum"]
    n_epochs = cross_val_args["n_epochs"]
    writer = cross_val_args["writer"]
    save_prefix = cross_val_args["save_prefix"]

    test_results = {}

    for n in num_hidden_layers:

        for sizes in product(*hidden_layer_sizes[-n:]):

            print(f"\n{n} hidden layers. hidden layer sizes: {sizes}\n")

            # set up model, loss function, optimization method
            model = MLP(D, K, n, sizes, use_dropout).to(device).double()
            print(model, '\n')
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            # train

            train_args = {
                "model": model,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
                "device": device,
                "n_epochs": n_epochs,
                "writer": writer,
                "save_prefix": save_prefix
            }

            train_epochs(data, **train_args)

            # evaluate on test set

            _, accuracy = eval_model(model, data, device, loss_fn)

            test_results[sizes] = accuracy
    
    for sizes in test_results:
        print(sizes, "test accuracy: %.4f" % test_results[sizes])
    print("\nBest model: {}".format(max(test_results, key=test_results.get)))

    return test_results
