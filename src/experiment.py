from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class Experiment:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            device: Optional[torch.device] = None
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = criterion
        self._optimizer = optimizer
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._model.to(self._device)

        self._train_loss = float("inf")
        self._val_loss = float("inf")

    def get_train_loss(self) -> float:
        return self._train_loss

    def get_val_loss(self) -> float:
        return self._val_loss

    @torch.no_grad()
    def validate_one_epoch(self) -> None:
        self._model.eval()
        validation_losses = []
        for (images, embeddings), labels in self._val_loader:

            images = images.to(self._device)
            embeddings = embeddings.to(self._device)
            labels = labels.to(self._device)

            predictions = self._model(images, embeddings)
            loss = self._criterion(predictions, labels.float())
            validation_losses.append(loss.item())
        self._val_loss = np.average(validation_losses)

    def train_one_epoch(self) -> None:
        self._model.train()
        train_losses = []
        for (images, embeddings), labels in self._train_loader:

            images = images.to(self._device)
            embeddings = embeddings.to(self._device)
            labels = labels.to(self._device)

            predictions = self._model(images, embeddings)
            loss = self._criterion(predictions, labels.float())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            train_losses.append(loss.item())
        self._train_loss = np.average(train_losses)

