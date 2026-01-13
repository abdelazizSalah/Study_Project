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
        '''
            Takes features vector as input and outputs real/fake score.
        '''
        super().__init__()

        # typical architecture for binary classification
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 512), # map the input feature vector to 512-dim
            nn.ReLU(), # add non-linearity

            nn.Linear(512, 256), # map to 256-dim
            nn.ReLU(), # add non-linearity for more complex decision boundary

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            # nn.Sigmoid() # output probability of being real (1) or fake (0)
            '''
                I removed the sigmoid here, because I read that during the training of GAN, the 
                D should output the raw logits not the probabilities, while the sigmoid should be used only during inference. 
                and this is because the logits are numerically more stable and easier to optimize. 

            '''
        )

    def forward(self, features):
        return self.fc_layers(features)
