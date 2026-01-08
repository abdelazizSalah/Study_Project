'''
@Author: Abdelaziz Neamatallah
@Date:17.12.25
@Description: This file contains the Generator model for GAN-based anomaly detection in ICS network traffic.
- We follow this logic:
    - Phase 3: Generator implementation
        - Noise input
            - We should define input as m*n also sampled from normal distribution
                - z = N(0,1)
        - Architecture:
            - 4 fully connected layers (LeakyReLU)
            - 8 convolutional layers (LeakyReLU)
            - Upsampling before first 4 conv layers
            - Final output layer to match (1,m,n) -> real packet shape
        - Output constraints:
            - Output range should match normalized packet range
            - No activation mismatches (e.g. sigmoid vs tanh confusion)
'''

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, m, n, kernel_size=3):
        super().__init__()

        self.m = m
        self.n = n
        self.z_dim = m * n

        # -------- Stage 1: Fully Connected layers --------
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),

            nn.Linear(2048, 256 * m * n),
            nn.LeakyReLU(0.2)
        )

        # -------- Stage 2 & 3: Convolutional layers --------
        self.conv_layers = nn.Sequential(
            # Upsampling + Conv (1)
            nn.Upsample(scale_factor=1),  # placeholder (keeps shape explicit)
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Upsampling + Conv (2)
            nn.Upsample(scale_factor=1),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Upsampling + Conv (3)
            nn.Upsample(scale_factor=1),
            nn.Conv2d(256, 128, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Upsampling + Conv (4)
            nn.Upsample(scale_factor=1),
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Conv (5)
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Conv (6)
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Conv (7)
            nn.Conv2d(32, 16, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),

            # Conv (8) – output layer
            nn.Conv2d(16, 1, kernel_size=kernel_size, padding=1),
            nn.Sigmoid()

            '''
            While here I kept the sigmoid, because the generator must always keep its output in range [0,1] as it output data not decision. 
            '''
        )

    def forward(self, z):
        """
        Input:
            z: noise vector of shape (batch_size, m*n)
        Output:
            generated packet tensor of shape (batch_size, 1, m, n)
        """
        x = self.fc(z)
        x = x.view(z.size(0), 256, self.m, self.n) # flatten
        x = self.conv_layers(x)
        return x
