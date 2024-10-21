import torch.nn as nn
import torch.nn.functional as F

class ImageNetMLP(nn.Module):
    def __init__(self):
        super(ImageNetMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(150528, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 4096)
        self.linear4 = nn.Linear(4096, 4096)
        self.linear5 = nn.Linear(4096, 4096)
        self.linear6 = nn.Linear(4096, 4096)
        self.linear7 = nn.Linear(4096, 4096)
        self.linear8 = nn.Linear(4096, 4096)
        self.linear9 = nn.Linear(4096, 4096)
        self.logit_layer = nn.Linear(4096, 1000)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.relu(self.linear6(x))
        x = self.relu(self.linear7(x))
        x = self.relu(self.linear8(x))
        x = self.relu(self.linear9(x))
        return self.logit_layer(x)

class ImageNetShallowMLP(nn.Module):
    def __init__(self):
        super(ImageNetShallowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(150528, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 1000),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

class ImageNetNarrowMLP(nn.Module):
    def __init__(self):
        super(ImageNetNarrowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.initial_layer = nn.Sequential(
            nn.Linear(150528, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.intermediate_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(2048 if i == 0 else 1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.1)
              ) for i in range(47)]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 1000),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.initial_layer(x)
        x = self.intermediate_layers(x)
        x = self.output_layer(x)
        return x