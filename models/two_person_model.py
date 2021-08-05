import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # class : friends, family, couple, professional, commercial, no_relation
        self.num_classes = 6

        # ResNet18
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # # ResNet 101
        # self.fc1 = nn.Linear(4096, 2048)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024, self.num_classes)

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)

        x = torch.cat((x1, x2), dim=1)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
