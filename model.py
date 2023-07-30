import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 512*512->256*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*256->128*128
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 256*256->128*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*128->64*64
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # 128*128->64*64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32*32
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # 32*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16*16
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512*16*16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.model(x)
