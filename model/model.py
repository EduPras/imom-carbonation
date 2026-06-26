import torch.nn as nn


class ConcretePredictor(nn.Module):
    def __init__(self):
        super(ConcretePredictor, self).__init__()

        # Input layer: 8 features -> 32 neurons
        self.fc1 = nn.Linear(8, 32)
        # Hidden layer: 32 neurons -> 16 neurons
        self.fc2 = nn.Linear(32, 16)
        # Output layer: 16 neurons -> 2 targets
        self.fc3 = nn.Linear(16, 2)

        # Activation and Regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
