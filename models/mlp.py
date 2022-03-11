from torch import nn


class MLP(nn.Module):

    def __init__(self, D, K, num_hidden_layers, num_perceptrons, use_dropout=False):
        super(MLP, self).__init__()

        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.25)

        self.fc_input = nn.Linear(D, num_perceptrons[0])
        self.fc_list = nn.ModuleList([
            nn.Linear(num_perceptrons[i-1], num_perceptrons[i]) 
            for i in range(1, num_hidden_layers)
        ])
        self.fc_output = nn.Linear(num_perceptrons[-1], K)

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