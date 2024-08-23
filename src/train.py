import logging
import random

import click
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import mlflow

from experiment import Experiment
from dataset import MsCocoDataset
from model import ContentCheckingModel

IMAGES_DIR = "data/processed/images"
EMBEDDINGS_FILE = "data/processed/embeddings.json"
DATASET_FILE = "data/processed/dataset.csv"


@click.command
@click.option("--batch-size", "-b", type=int)
@click.option("--epochs", "-e", type=int)
@click.option("--learning-rate", "--lr", "-l", type=float)
def main(batch_size: int, epochs: int, learning_rate: float):
    model = ContentCheckingModel()
    train_loader, val_loader = get_data_loaders(batch_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    logger = logging.getLogger()

    mlflow.pytorch.autolog()
    experiment = Experiment(model, train_loader, val_loader, criterion, optimizer)
    with mlflow.start_run() as run:
        for i in range(epochs):
            experiment.train_one_epoch()
            experiment.validate_one_epoch()
            logger.log(mlflow.pytorch.get_run())


def get_data_loaders(batch_size):
    random.seed = 0
    torch.manual_seed(0)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = MsCocoDataset(IMAGES_DIR, EMBEDDINGS_FILE, DATASET_FILE, image_transform)
    train_set, val_set, _ = random_split(dataset, (.7, .15, .15))
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == '__main__':
    main()
