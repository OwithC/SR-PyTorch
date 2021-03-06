import torch
import torch.nn as nn
import torchvision.models as models


class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # class : friends, family, couple, professional, commercial, no_relation
        self.num_classes = 6

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, self.num_classes, bias=True)
        )
        
    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        
        x = torch.cat((x1, x2), dim=1)

        x = self.fc(x)

        return x
    
    
class vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # class : friends, family, couple, professional, commercial, no_relation
        self.num_classes = 6

        self.fc = nn.Sequential(
            nn.Linear(50176, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, self.num_classes, bias=True)
        )

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        
        x = torch.cat((x1, x2), dim=1)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x
