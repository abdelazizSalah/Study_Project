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
        '''
            Input shape convention is: (channels, height, width)
                - channels = 1 for bytes vector
                - height = # of packets -> m = 10
                - width = # of bytes per packet -> n (varying)
        '''
        
        super().__init__()

        # extract the falttened feature vector per sample (batch_size, feature_dim)
        self.feature_extractor = DiscriminatorFeatures()

        # Compute feature dimension dynamically
        with torch.no_grad(): # disable gradient calculation because this step is not part of training
            '''
            The main goal here is to probe the network and determine 
            the size of the output feature vector after passing an input through the feature extractor.
            
            *input_shape means to unpack the tuple, which mean that if the input_shape was (1,m,n)
            it is converted to (1,1,m,n) to be compatible with the pytourch shape. 
            '''

            # create fake input tensor filled with zeros
            dummy = torch.zeros(1, *input_shape) # 1 is batch size, *input_shape unpacks the input shape tuple

            # runs this dummy input through the feature extractor to get the feature dimension
            feature_dim = self.feature_extractor(dummy).shape[1] # number of features per sample. 

        # define a classifier using the computed feature dimension
        self.classifier = DiscriminatorClassifier(feature_dim)

    def extract_features(self, x):
        '''
            - This function to extract the features part only, which will be used in feature matching loss.
        '''
        return self.feature_extractor(x)

    def forward(self, x):
        '''
            - This function is to perform the full forward pass, which will be used in real/fake classification.
        '''
        features = self.extract_features(x)
        score = self.classifier(features)
        return score
