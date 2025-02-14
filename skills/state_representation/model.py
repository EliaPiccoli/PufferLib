import torch as th
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return th.reshape(x, (x.size(0), -1))

class NatureCNN(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.input_channels = input_channels
        self.final_conv_size = 64 * 9 * 6

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(self.final_conv_size, self.feature_size),
            #nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)