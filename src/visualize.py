import numpy as np
import torch
from matplotlib import pyplot as plt

def show_images(images, labels, preds, class_names):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)

        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'

        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()


def show_preds(model, dl_test, class_names):
    model.eval()
    images, labels = next(iter(dl_test))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds, class_names)
