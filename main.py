import torch
from src.dataset import ChestXRayDataset
from src.transforms import train_transform, test_transform
from src.model import build_model
from src.train import train
from src.visualize import show_images, show_preds

torch.manual_seed(0)
ROOT_DIR = 'COVID-19 Radiography Database'

BATCH_SIZE = 6
EPOCHS = 1
def main():
    train_dirs = {
        'normal': f'{ROOT_DIR}/normal',
        'viral': f'{ROOT_DIR}/viral',
        'covid': f'{ROOT_DIR}/covid'
    }
    test_dirs = {
        'normal': f'{ROOT_DIR}/test/normal',
        'viral': f'{ROOT_DIR}/test/viral',
        'covid': f'{ROOT_DIR}/test/covid'
    }
    train_dataset = ChestXRayDataset(train_dirs, train_transform)
    test_dataset = ChestXRayDataset(test_dirs, test_transform)
    class_names = train_dataset.class_names
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Number of training batches', len(dl_train))
    print('Number of test batches', len(dl_test))
    images, labels = next(iter(dl_train))
    show_images(images, labels, labels, class_names)
    images, labels = next(iter(dl_test))
    show_images(images, labels, labels, class_names)
    model, loss_fn, optimizer = build_model(num_classes=3, learning_rate=3e-5)
    show_preds(model, dl_test, class_names)
    train(model, dl_train, dl_test, loss_fn, optimizer, test_dataset,
          class_names, epochs=EPOCHS)
    show_preds(model, dl_test, class_names)


if __name__ == '__main__':
    main()
