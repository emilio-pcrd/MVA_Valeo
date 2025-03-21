import torch.nn as nn
import timm
from torchvision.models import resnet34, resnet101


class Classifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Classifier, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)


class Finetuned_Resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(Finetuned_Resnet34, self).__init__()
        model = resnet34(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes),
        )
        self.resnet = model

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.resnet(x)
        return output


class Finetuned_Resnet101(nn.Module):
    def __init__(self, num_classes=6):
        super(Finetuned_Resnet101, self).__init__()
        model = resnet101(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes),
        )
        self.resnet = model

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.resnet(x)
        return output
