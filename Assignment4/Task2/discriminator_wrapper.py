'''
@Author: Abdelaziz Neamatallah
@Date:17.12.25
@Description: This file contains the Discriminator wrapper model for GAN-based anomaly detection in ICS network traffic.
- This model combines the Discriminator feature extractor and classifier.
'''

import torch
import torch.nn as nn
from  discriminator_classifier import DiscriminatorClassifier
from discriminator_feature_extractor import DiscriminatorFeatures



class Discriminator(nn.Module):
    '''
        - The main purpose of this class is to wrap the Discriminator feature extractor and classifier into a single model.
        - So we can call D.extract_features(x) to get features which will be used in feature matching loss.
        - And we can call D(x) to get the final real/fake score.
    '''
    def __init__(self, input_shape):
        super().__init__()

        self.feature_extractor = DiscriminatorFeatures()

        # Compute feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feature_dim = self.feature_extractor(dummy).shape[1]

        self.classifier = DiscriminatorClassifier(feature_dim)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.extract_features(x)
        score = self.classifier(features)
        return score
