import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, feat_in, n_layers, n_units, prob_drop):
        super().__init__()

        self.feat_in = feat_in
        self.n_layers = n_layers
        self.n_units = n_units
        self.prob_drop = prob_drop
        self.dropout = nn.Dropout(prob_drop)

        # use ModuleList to store the layers
        # to be iterated over in the forward loop
        self.fc = nn.ModuleList()

        # create each layer and append to ModuleList
        for i in range(n_layers):
            if i == 0:
                self.fc.append(nn.Linear(feat_in, n_units))
            elif i == n_layers - 1:
                self.fc.append(nn.Linear(n_units, 1))
            else:
                self.fc.append(nn.Linear(n_units, n_units))

    def forward(self, x):
        # iterate through each layer in the forward pass
        # and apply the appropriate activation function
        for i, layer in enumerate(self.fc):
            if i == 0:
                # if input layer, do not apply any acvitvation
                # but do apply a dropout
                x = layer(x)
                x = self.dropout(x)

            elif i == self.n_layers - 1:
                # sigmoid in last layer so that
                # output is between 0 and 1
                x = torch.sigmoid(layer(x))

            else:
                # hidden layer
                x = torch.relu(layer(x))
                x = self.dropout(x)
        return x

