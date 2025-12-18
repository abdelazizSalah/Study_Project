'''
@Author: Abdelaziz Neamatallah
@Date:17.12.25
@Description: This file contains the Discriminator classifier model for GAN-based anomaly detection in ICS network traffic.
- This model decides real vs fake based on the extracted features from the Discriminator feature extractor.
- Design assumptions:
    - 4 fully connected layers
    - ReLU activation function everywhere except the last layer
    - final sigmoid activation to output probability -> [0, 1]
    

'''

import torch
import torch.nn as nn


class DiscriminatorClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.fc_layers(features)
