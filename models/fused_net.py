from torch import nn
from .mlp import MLP


class FusedNet(nn.Module):

    def __init__(self, 
                 use_models, 
                 n_classes, 
                 num_hidden_layers, 
                 num_perceptrons, 
                 use_dropout=False
                ):
        # num perceptrons is a list containing number of perceptrons per 
        # hidden layer
        super(FusedNet, self).__init__()
        n_models = 0
        for model in use_models:
            if use_models[model]:
                n_models += 1
        assert n_models > 0, "Must use at least one model"

        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

        self.fc_input = nn.Linear(n_models * n_classes, num_perceptrons[0])
        self.fc_list = nn.ModuleList([
            nn.Linear(num_perceptrons[i-1], num_perceptrons[i]) 
            for i in range(1, num_hidden_layers)
        ])
        self.fc_output = nn.Linear(num_perceptrons[-1], n_classes)
    
    def forward(self, x):
        x = self.fc_input(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(x)

        for linear in self.fc_list:
            x = linear(x)
            if self.use_dropout:
                x = self.dropout(x)
            x = self.relu(x)
        
        x = self.fc_output(x)
        return x