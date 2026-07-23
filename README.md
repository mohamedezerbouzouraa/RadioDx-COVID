# COVID-19 Chest X-Ray Classifier

Deep learning model classifying chest X-rays into Normal, Viral Pneumonia, and COVID-19 using PyTorch and transfer learning (ResNet18). Includes data preprocessing, a custom Dataset class, a training loop with early stopping, and prediction visualization.

## Dataset

This project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle. Download and extract it into the project root so it matches this structure:

```
COVID-19 Radiography Database/
├── NORMAL/
├── Viral Pneumonia/
└── COVID-19/
```

## Project Structure

```
covid-xray-classifier/
├── data/
│   └── prepare_data.py   # Reorganizes raw dataset into train/test folders
├── src/
│   ├── dataset.py         # Custom PyTorch Dataset class
│   ├── transforms.py      # Image preprocessing/augmentation pipelines
│   ├── model.py            # ResNet18 model, loss, optimizer setup
│   ├── train.py             # Training loop with validation
│   └── visualize.py        # Functions to display samples/predictions
├── main.py                  # Entry point tying everything together
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Prepare the dataset** (run once, after downloading and extracting the raw dataset):

```bash
python data/prepare_data.py
```

This renames the raw folders to `normal`, `viral`, `covid`, and moves 30 images per class into a `test/` subfolder.

**2. Train the model:**

```bash
python main.py
```

This loads the data, builds a ResNet18 model pretrained on ImageNet with a modified final layer for 3-class output, trains it, and displays sample predictions (correct predictions in green, incorrect in red).

## How It Works

- **Transfer learning**: A ResNet18 pretrained on ImageNet is used as a feature extractor, with only the final layer retrained for this 3-class problem.
- **Class balancing**: The custom `ChestXRayDataset` randomly samples a class before selecting an image, giving each class roughly equal representation during training regardless of dataset imbalance.
- **Data augmentation**: Random horizontal flipping is applied during training only, to improve generalization.
- **Early stopping**: Training automatically stops once validation accuracy reaches 95%.

## License

Dataset files © original authors (see Kaggle page for details).
