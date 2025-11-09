'''
@Author Abdelaziz Neamatallah
@Date 09.11.25
@Description Stacked Denoising Autoencoder implementation using Keras
'''

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as Keras
from keras.utils.vis_utils import plot_model
import scipy.sparse as scp


class SDA(object) : 
    '''
      This class implements Stacked Denoising Autoencoder using Keras library.
    '''
    def __init__(self, 
                numLayers= 1,
                hiddenNodesPerLayer = [32],
                dropoutPerLayer = [0.1],
                layerType = "dense",
                activationType = "relu", 
                
                encodingActivationPerLayer = ['relu'],
                decodingActivationPerLayer = ['sigmoid'],
                bias = True,
                lossFunction = 'mse',
                batchSize = 128,
                numberOfEpochs = 100,
                optimizer = 'adam', 
                noiseFactor = 0.1,
                  ):
        '''
            Initializing SDA parameters
            @param numLayers: number of autoencoders to stack on top of each other.
            @param hiddenNodesPerLayer: list of number of hidden nodes for each layer.
            @param dropoutPerLayer: list of dropout values for each layer.
            @param encodingActivationPerLayer: list of activation functions for encoding part of each layer.
            @param decodingActivationPerLayer: list of activation functions for decoding part of each layer.
            @param bias: whether to use bias in each layer or not.
            @param lossFunction: loss function to use during training.
            @param batchSize: number of samples per gradient update. which determines the number of samples to work through before updating the internal model parameters.
            @param numberOfEpochs: number of epochs to train the model. which is one complete pass through the whole training dataset.
            @param optimizer: optimizer to use during training.
            @param noiseFactor: fraction of input to corrupt with noise during training.
            @param counter: counter to differentiate between multiple SDA instances.
        '''
        self.numLayers = numLayers
        self.hiddenNodesPerLayer, self.dropoutPerLayer, self.encodingActivationPerLayer, self.decodingActivationPerLayer = self._assertInputs(numLayers, hiddenNodesPerLayer, dropoutPerLayer, encodingActivationPerLayer, decodingActivationPerLayer)
        self.bias = bias
        self.lossFunction = lossFunction
        self.batchSize = batchSize
        self.numberOfEpochs = numberOfEpochs
        self.optimizer = optimizer
        self.noiseFactor = noiseFactor
        self.layerType = layerType
        self.activationType = activationType
    
    def getSDAModel(self, trainingData, validationData, testingData, outputDirectory = 'models_and_data/' ):
        '''
            This function creates and trains the Stacked Denoising Autoencoder model.
            Each layer is trained on at a time, to allow leaning stable meaningful features step by step, and avoid vanishing gradients.
            Also to build hierarichal understanding, so each layer learn simple features, and then the next combines the previous features to learn more complex features.
            Which helps in improving the convergence of the model. 

            After all layers are trained, we stack all encoder layers together on top of each other along with the dropout layers, 
            to create a Sequential model that passes the data through all trained encoder layers.
            returns the stacked model along with the encoded feature vectors for our training, validation, and test data.

            @param trainingData: training data as numpy array.
            @param validationData: validation data as numpy array.
            @param testingData: testing data as numpy array.
            @param outputDirectory: directory to save the trained models.
            @return: stacked SDA model, dense representation of training, validation, and test data.
        '''

        reconstructionMSE = 0
        modelLayers = [] # list to hold each layer model after training it.
        encoders = []
        # add noise to the input following gaussian distribution
        # TODO: Add the noise.
        for currLayer in range(self.numLayers): 
            # start creating the SDA 
            inputLayer = Input(shape = (trainingData.shape[1],)) # the input layer should contains the number of features in the data. M (packet size)
            
            # specifing the dropout layer.
            dropOutLayer = Dropout(self.dropoutPerLayer[currLayer])
            inputAfterDropout = dropOutLayer(inputLayer)

            # creating the encoding Dense layers with the given parameters 
            encodingLayer = Dense(output_dim = self.hiddenNodesPerLayer[currLayer], 
                                  init = 'glorot_uniform', # initializes weights efficiently.
                                  activation = self.encodingActivationPerLayer[currLayer],
                                  bias = self.bias,
                                  name = f'encodingLayer{str(currLayer)}_{self.layerType}_{self.activationType}' 
                                   )
            encoder = encodingLayer(inputAfterDropout)

            # creating the decoding Dense layers with the given parameters
            numberOfOutputNodes = trainingData.shape[1] # same output units as the input units. 
            decodingLayer = Dense(output_dim = numberOfOutputNodes,
                                  init = 'glorot_uniform',
                                  activation = self.decodingActivationPerLayer[currLayer],
                                   bias = self.bias,
                                  name = f'decodingLayer{str(currLayer)}_{self.layerType}_{self.activationType}'
                                   )
            decoder = decodingLayer(encoder)

            # creaing keras model 
            currentModel = Model(inputLayer,decoder)
            
            # compiling the model with the given loss function and optimizer
            currentModel.compile(optimizer = self.optimizer, loss = self.lossFunction)

            # defining early stopping callback to prevent overfitting
            earlyStoppingCallback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # training the current layer model using mini-batch 
            print('Before training layer ', str(currLayer))
            currentModel.fit_generator(
                generator = self._batchGeneratorForTraining(
                            trainingData, trainingData, # input and outputs are the same, because autoencoder generates the input
                            self.batchSize, # batch size
                            shuffle = True # change the order to prevent bias.
                            ),
                nb_epoch = self.numberOfEpochs,
                samples_per_epoch = trainingData.shape[0], # the whole dataset should be passed in each epoch.
                callbacks = [earlyStoppingCallback], # function run after each epoch
                validation_data = self._batchGeneratorForTraining(
                                validationData, validationData,
                                self.batchSize,
                                shuffle = False # because it shouldn't impact the validation.
                            ),
                nb_val_samples = validationData.shape[0], # the whole validation dataset
                validation_steps = np.ceil(validationData.shape[0]/self.batchSize) # validation occurs after each batch
                          )
            print('After training layer ', str(currLayer))
            
            # save the trained model in the list
            modelLayers.append(currentModel)

            # extract its encoder part only for stacking later
            encoderLayer = currentModel.layers[-2] # because the last layer is the decoder layer

            # add it tp the list
            encoders.append(encoderLayer)

            # compute the mse on the first layer only, because it is the one with the same input as the data
            if currLayer == 0:                
                # compute the mean square error
                reconstructionMSE = self._getReconstructionError(currentModel, trainingData, numberOfNeurons = currentModel.layers[-1].output_shape[1]) # uses the output size of the decoder.

            # Extracting the hidden layer output (the encoded representation) from the trained moder, 
            trainingData = self._getIntermediateLayerOutput(currentModel, trainingData, layerNumber = 2, applyDropOut = 0, numberOfNeurons = self.hiddenNodesPerLayer[currLayer], batchSize = self.batchSize) #train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of second layer: hidden layer
            assert trainingData.shape[1] == self.hiddenNodesPerLayer[currLayer], "Output of hidden layer not retrieved"
            validationData = self._getIntermediateLayerOutput(currentModel, validationData, layerNumber = 2, applyDropOut = 0, numberOfNeurons = self.hiddenNodesPerLayer[currLayer], batchSize = self.batchSize) #get output of second layer (hidden layer) without dropout
            testingData = self._getIntermediateLayerOutput(currentModel, testingData, layerNumber = 2, applyDropOut = 0, numberOfNeurons = self.hiddenNodesPerLayer[currLayer], batchSize = self.batchSize)
            
        # Writing all configurations to file 
        self._writeSDAConfigToFile(outputDirectory)

        # creating the full stacked sequential auto encoder model after training all layers
        finalModel = self._buildModelFromEncoders(encoders, dropoutAll = True)

        # Saving the file into the directory. 
        self._saveModel(finalModel, outputDir = outputDirectory, architectureFileName = f'enc_{self.layerType}_{self.activationType}_layers.png', modelJsonFileName = f'enc_{self.layerType}_{self.activationType}_layers.json', weightsFileName = 'enc_layers_weights.h5')
        return finalModel, trainingData, validationData, testingData, reconstructionMSE
    
    # This saves the trained model
    def _saveModel(model, outputDir, architectureFileName = 'model_arch.png', modelJsonFileName = 'model_arch.json', weightsFileName = 'model_weights.h5'):
        '''
        Saves a Keras model description and model weights
        @param model: a keras model
        @param outputDir: directory to save model architecture and weights to
        @param modelJsonFileName: file name for model architecture
        @param weightsFileName: filename for model weights
        '''
        model.summary() # prints the model structure. 
        plot_model(model, to_file=outputDir+architectureFileName) # provides a visual diagram for the model. 
        
        # exporting the architecture to a json file
        jsonString = model.to_json()
        open(outputDir+modelJsonFileName, 'w').write(jsonString)
        
        # and saving its weights.
        model.save_weights(outputDir+weightsFileName, overwrite=True)
        
    # This stores the SDA hyperparameters to a file for future reference.
    def _writeSDAConfigToFile(self, outDir):
        """
        Write the configuration of the autoencoder to a file
        """
        with open(outDir + f'sdae_config_{self.layerType}_{self.activationType}.txt', 'w') as f:
            f.write("Number of layers: " + str(self.numLayers))
            f.write("\nHidden nodes: ")
            for i in range(self.numLayers):
                f.write(str(self.hiddenNodesPerLayer[i])+' ')
                
            f.write("\nDropout: ")
            for i in range(self.numLayers):
                f.write(str(self.dropoutPerLayer[i])+' ')
            
            f.write("\nEncoder activation: ")
            for i in range(self.numLayers):
                f.write(str(self.encodingActivationPerLayer[i])+' ')
                
            f.write("\nDecoder activation: ")
            for i in range(self.numLayers):
                f.write(str(self.decodingActivationPerLayer[i])+' ')
            
            f.write("\nEpochs: " + str(self.numberOfEpochs))
            
            f.write("\nBias: " + str(self.bias))
            f.write("\nLoss: " + str(self.lossFunction))
            f.write("\nBatch size: " + str(self.batchSize))
            f.write("\nOptimizer: " + str(self.optimizer))

            
    def _buildModelFromEncoders(self, encodingLayers, dropoutAll = False ):
            '''
            Builds a deep NN model that generates low-dimensional representation of input, based on pretrained layers.
            @param encodingLayers: pretrained encoder layers
            @param dropoutAll: True to include dropout layer between all layers. By default, dropout is only present for input.
            @return model with each encoding layer as a layer of a NN
            '''
            model = Sequential()
            model.add(Dropout(self.dropout[0], input_shape = (encodingLayers[0].input_shape[1],)))
            
            for i in range(len(encodingLayers)):
                if i and dropoutAll: # insert the dropout between all layers except the first one, because we already did.
                    model.add(Dropout(self.dropout[i]))
                    
                encodingLayers[i].inbound_nodes = [] # remove previous connections to avoid errors
                model.add(encodingLayers[i]) # add the encoder layer
            
            return model
    
    
    # while this function adds logic to handle larger datasets, and converting sparse to dense in batches, then uses the getNthLayer.
    def _getIntermediateLayerOutput(self, model, inputData, layerNumber, applyDropOut, numberOfNeurons, batchSize, dtype = np.float32):
        '''
        Returns output of a given intermediate layer in a model
        @param model: model to get output from
        @param inputData: sparse representation of input data
        @param layerNumber: the layer number for which output is required
        @param applyDropOut: (0/1) 1 to use training config, like dropout noise.
        @param numberOfNeurons: number of output nodes in the given layer (pre-specify so as to use generator function with sparse matrix to get layer output)
        @param batchSize: the num of instances to convert to dense at a time
        @return value of intermediate layer
        '''

        # creating empty numpy array to hold output data
        outputData = np.zeros(shape = (inputData.shape[0],numberOfNeurons)) # number of instances * number of neurons.
        

        generatedBatch = self._batchGeneratorForExtraction(inputData, batchSize = batchSize)
        stopIter = int(np.ceil(inputData.shape[0]/batchSize))
        
        for _ in range(stopIter): 
            curBatch, curBatchIdx = next(generatedBatch) # this is how to handle yield, and get the next batch with its indicies.
            outputData[curBatchIdx,:] = self._getNthLayerOutput(model, layerNumber, inputData =  curBatch, isTraining = applyDropOut)
        
        return outputData.astype(dtype, copy = False)


    # This communicate with keras backend function to compute the output of nth layer
    def _getNthLayerOutput(self, model, layerNumber, inputData, isTraining = 1):
        '''
        Returns output of nth layer in a given model.
        @param model: keras model to get an intermediate value out of
        @param layerNumber: the layer number to get the value of
        @param inputData: input data for which layer value should be computed and returned.
        @param isTraining: (1/0): 1 to use the same setting as training (for example, with Dropout, etc.), 0 to use the same setting as testing phase for the model.
        @return the value of layerNumber in the given model, input, and setting 
        '''
        # This is a keras backend function that allows me to create a callable function that directly computes specific layer output.
        # 
        getNthLayerOutput = Keras.function(
            [model.layers[0].input,# specifing model's input tensor
            Keras.learning_phase()],# tells Keras whether the model in in training mode or inference mode
            [model.layers[layerNumber].output] # specifing which layer output to return
              )
        # So the main idea is to be able to extract the output of any layer.
        return getNthLayerOutput([inputData,isTraining])[0] # calling the function with input data and training/inference mode, and returning the output.
            
    

    def _getReconstructionError(self, model, inputData, numberOfNeurons):
        """
        Return reconstruction squared error at individual nodes, averaged across all instances.
        @param model: trained model
        @param inputData: input data to reconstruct
        @param numberOfNeurons: number of model output nodes
        """
        # -1 is the last layer
        trainReconstruction = self._getIntermediateLayerOutput(model, inputData, layerNumber = -1, isTraining = 0, numberOfNeurons = numberOfNeurons, batchSize = self.batchSize) #train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of third layer: output layer
        
        # mean error
        reconMSE = np.mean(np.square(trainReconstruction - inputData), axis = 0)
        
        reconMSE = np.ravel(reconMSE)
        
        return reconMSE

    def _batchGeneratorForExtraction(X, batchSize):
        '''
        Creates batches of data from given input, given a batch size. Returns dense representation of sparse input one batch a time.
        @param X: input features, can be sparse or dense
        @param batchSize: number of instances in each batch
        @return batch of input data
        '''
        # determine how many batches are needed
        numberOfBatches = np.ceil(X.shape[0]/batchSize) #ceil function allows for creating last batch off remaining samples

        # get evently spaced indices from 0 to number of instances.
        sampleIndex = np.arange(X.shape[0]) 
        
        # check if the given matrix is sparse or dense
        sparse = False
        if scp.issparse(X):
            sparse = True
            
        
        counter = 0
        while counter < numberOfBatches: 
            batchIndex = sampleIndex[batchSize*counter:batchSize*(counter+1)]
            if sparse:
                x_batch = X[batchIndex,:].toarray() #converts to dense array
            else:
                x_batch = X[batchIndex,:] # this to ensure that each batch given to the model is dense.
            yield x_batch, batchIndex # return the current batch and its original row indicies, and when called again, it will continue from here.
            counter += 1 # this is how yield is differnt from return, it keeps the state of the function, and we handle it with (next) function



    def _batchGeneratorForTraining(X, Y, batch_size, shuffle, seed = 1337):
        '''
        Creates batches of data from given dataset, given a batch size. Returns dense representation of sparse input.
        @param X: input features, sparse or dense
        @param Y: input labels, sparse or dense
        @param batch_size: number of instances in each batch
        @param shuffle: If True, shuffle input instances.
        @param seed: fixed seed for shuffling data, for replication
        @return batch of input features and labels
        '''
        number_of_batches = np.ceil(X.shape[0]/batch_size) #ceil function allows for creating last batch off remaining samples
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(sample_index)
        
        sparse = False
        if scp.issparse(X):
            sparse = True
        
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            if sparse:
                x_batch = X[batch_index,:].toarray() #converts to dense array
                y_batch = Y[batch_index,:].toarray() #converts to dense array
            else:
                x_batch = X[batch_index,:]
                y_batch = Y[batch_index,:]
            counter += 1
            yield x_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0



    def _assertInputs(self, numLayers, hiddenNodesPerLayer, dropoutPerLayer, encodingActivationPerLayer, decodingActivationPerLayer):
        '''
            Assert that the inputs are valid.
        '''
        if len(hiddenNodesPerLayer) ==1: 
            hiddenNodesPerLayer = hiddenNodesPerLayer * numLayers
        if len(dropoutPerLayer) ==1:
            dropoutPerLayer = dropoutPerLayer * numLayers
        if len(encodingActivationPerLayer) ==1:
            encodingActivationPerLayer = encodingActivationPerLayer * numLayers
        if len(decodingActivationPerLayer) ==1:
            decodingActivationPerLayer = decodingActivationPerLayer * numLayers
        assert len(hiddenNodesPerLayer) == numLayers, "Length of hiddenNodesPerLayer must be equal to numLayers"
        assert len(dropoutPerLayer) == numLayers, "Length of dropoutPerLayer must be equal to numLayers"
        assert len(encodingActivationPerLayer) == numLayers, "Length of encodingActivationPerLayer must be equal to numLayers"
        assert len(decodingActivationPerLayer) == numLayers, "Length of decodingActivationPerLayer must be equal to numLayers"
        return hiddenNodesPerLayer, dropoutPerLayer, encodingActivationPerLayer, decodingActivationPerLayer