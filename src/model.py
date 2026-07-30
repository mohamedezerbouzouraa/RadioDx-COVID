import torch
import torchvision


def build_model(num_classes=3, learning_rate=3e-5):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer
