'''
@Author: Abdelaziz Neamatallah
@Date:17.12.25
@Description: This file contains the Discriminator feature extractor model for GAN-based anomaly detection in ICS network traffic.
- Design assumptions: 
    - Input shape convention is: (batch_size, channels, height, width)
        - batch_size = number of samples in a batch (how many samples we feed to the model at once).
        - channels = 1 for bytes vector
        - height = # of packets -> m = 10
        - width = # of bytes per packet -> n (varying)
    - We use Conv2D
    - We use ReLU activation function everywhere
    - Dropout after conv layers except the first and last conv layers
- Architecture: 
    - 9 convolutional layers with increasing number of filters
    - ReLU everywhere
    - Dropout everywhere except after the first and last conv layers
    - Output: flattened feature vector
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorFeatures(nn.Module):
    def __init__(self, dropout_p=0.3, kernel_size=3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Conv 1 (NO dropout)
            # This learns Byte primitives: 
            # e.g. byte boundaries, small byte value changes, noise vs structure
            nn.Conv2d(1, 16, kernel_size=kernel_size, padding=1), # 1 input channel -> 16 output channels (learning 16 different filters)
            nn.ReLU(),

            # Conv 2
            # Here it tries to learn local packet patterns
            # e.g. small opcodes, local packet structures, repeated byte motifs
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=1), # channels increase as depth increases to follow hierarchical representation principle
                                                                   # deeper layers should learn more complex features.
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 3
            # Intra packet structures
            # e.g. common header fields, protocol-specific patterns
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 4
            # we keep the same channels here to allow learning more complex intra-packet features
            # kind of repeating the same learning step to refine the learned features
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 5
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 6
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 7
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 8
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Conv 9 (NO dropout)
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

    def forward(self, x):
        '''
        Input:
            - x: input tensor of shape (batch_size, 1, m, n)
        Output:
            - x: flattened feature vector from the conv layers
        Logic:
            - Pass the input through the convolutional layers -> forward pass
            - Flatten the output to create a feature vector
        '''
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
