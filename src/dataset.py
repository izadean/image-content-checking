import os
import json
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import PIL.Image as Image


class MsCocoDataset(Dataset):
    def __init__(self, images_dir, embeddings_file, dataset_file, transform) -> None:
        super(MsCocoDataset, self).__init__()
        self.images_dir = images_dir
        with open(embeddings_file, "r") as f:
            self.embeddings = json.load(f)
        self.dataset = pd.read_csv(dataset_file)
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        record = self.dataset.loc[index, :]
        image_path = os.path.join(self.images_dir, record["file_name"])
        try:
            image = self.transform(Image.open(image_path))
            if image.shape[0] != 3:
                raise FileNotFoundError()
        except FileNotFoundError:
            print(f"WARNING: Can't find image with path '{image_path}'")
            return self[index+5]
        embedding = torch.tensor(self.embeddings[str(record["embeddings_id"])])
        label = F.one_hot(torch.tensor(record["label"]), num_classes=2)
        return (image, embedding), label

