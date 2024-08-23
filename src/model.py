import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights


class ContentCheckingModel(nn.Module):
    def __init__(self):
        super(ContentCheckingModel, self).__init__()

        self.image_encoder = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        self.image_encoder.heads = nn.Identity()

        self.final_net = nn.Sequential()
        in_features = 1152
        while in_features > 10:
            self.final_net.append(nn.Linear(in_features, in_features//2))
            self.final_net.append(nn.LeakyReLU())
            in_features = in_features // 2
        self.final_net.append(nn.Linear(in_features, 2))
        self.final_net.append(nn.Sigmoid())

    def forward(self, images, embeddings):
        images_features = self.image_encoder(images)
        result = self.final_net(torch.concat([images_features, embeddings], dim=1))
        return result
