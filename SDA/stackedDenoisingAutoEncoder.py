'''
@Author Abdelaziz Neamatallah
@Date 09.11.25
@Description Stacked Denoising Autoencoder implementation using Keras
'''


import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, EinsumDense

from keras.callbacks import EarlyStopping
from keras import backend as Keras
from keras.utils import plot_model
import scipy.sparse as scp

import tensorflow as tf

class SDA(object) : 
    '''
      This class implements Stacked Denoising Autoencoder using Keras library.
    '''
    def __init__(self, 
                numLayers= 1,
                hiddenNodesPerLayer = [32],
                dropoutPerLayer = [0.1],
                layerType = "dense",
                activationType = "relu", # for print and saving purposes
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
        print('initializing SDA parameters')
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
    
    def getSDAModel(self, trainingData, validationData, testingData,  outputDirectory = 'models_and_data/' ):
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
        noisyTrainingData = trainingData + self.noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=trainingData.shape)
        noisyValidationData = validationData + self.noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=validationData.shape)
        layer = self.layerType.capitalize() # to match the class names 
        originalTrainingData = trainingData
        originalNoisyTrainingData = noisyTrainingData
        originalValidationData = validationData
        originalNoisyValidationData = noisyValidationData
        for currLayer in range(self.numLayers): 
            print('Starting to train SDA with ', str(self.numLayers), ' layers of type ', layer)
            # start creating the SDA 
            inputLayer = Input(shape = (trainingData.shape[1],)) # the input layer should contains the number of features in the data. M (packet size)
            
            # specifing the dropout layer.
            dropOutLayer = Dropout(self.dropoutPerLayer[currLayer])
            inputAfterDropout = dropOutLayer(inputLayer)

            # creating the encoding Dense layers with the given parameters 
            if layer == 'Dense':
                print('Creating Dense layer ', str(currLayer))
                encodingLayer = Dense(units = self.hiddenNodesPerLayer[currLayer], 
                                    kernel_initializer = 'glorot_uniform', # initializes weights efficiently.
                                    activation = self.encodingActivationPerLayer[currLayer],
                                    use_bias = self.bias,
                                    name = f'encodingLayer{str(currLayer)}_{self.layerType}_{self.activationType}' 
                                    )
                encoder = encodingLayer(inputAfterDropout)

                # creating the decoding Dense layers with the given parameters
                numberOfOutputNodes = trainingData.shape[1] # same output units as the input units. 
                decodingLayer = Dense(units = numberOfOutputNodes,
                                    kernel_initializer = 'glorot_uniform',
                                    activation = self.decodingActivationPerLayer[currLayer],
                                    use_bias = self.bias,
                                    name = f'decodingLayer{str(currLayer)}_{self.layerType}_{self.activationType}'
                                    )
                decoder = decodingLayer(encoder)
            elif layer == 'Einsum':
                                
                encodingLayer = EinsumDense(
                    equation="ab,bc->ac",              # standard dense pattern
                    output_shape=(self.hiddenNodesPerLayer[currLayer],),
                    activation=self.encodingActivationPerLayer[currLayer],
                    bias_axes="c" if self.bias else None,
                    kernel_initializer='glorot_uniform',
                    name=f"encodingEinsumLayer{currLayer}_{self.layerType}_{self.activationType}"
                )
                encoder = encodingLayer(inputAfterDropout)
                decodingLayer = EinsumDense(
                    equation="ab,bc->ac",
                    output_shape=(trainingData.shape[1],),
                    activation=self.decodingActivationPerLayer[currLayer],
                    bias_axes="c" if self.bias else None,
                    kernel_initializer='glorot_uniform',
                    name=f"decodingEinsumLayer{currLayer}_{self.layerType}_{self.activationType}"
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
            currentModel.fit(
                self._batchGeneratorForTraining(
                    noisyTrainingData, trainingData,
                    self.batchSize,
                    shuffle=True
                ),
                epochs=self.numberOfEpochs,
                steps_per_epoch=int(np.ceil(noisyTrainingData.shape[0] / self.batchSize)),
                callbacks=[earlyStoppingCallback],
                validation_data=self._batchGeneratorForTraining(
                    noisyValidationData, validationData,
                    self.batchSize,
                    shuffle=False
                ),
                validation_steps=int(np.ceil(noisyValidationData.shape[0] / self.batchSize))
            )
            print('After training layer ', str(currLayer))
            
            # compute the mse on the first layer only, because it is the one with the same input as the data
            if currLayer == 0:                
                # compute the mean square errorR'..
                reconstructionMSE = self._getReconstructionError(currentModel, trainingData, numberOfNeurons = currentModel.output_shape[1]) # uses the output size of the decoder.
                
            # Save trained model
            modelLayers.append(currentModel)

            # Extract encoder model (input → encoded)
            encoderModel = Model(inputs=currentModel.input, outputs=currentModel.layers[-2].output)
            encoders.append(encoderModel)

            # Update training/validation/testing data to use this encoded representation
            trainingData = encoderModel.predict(trainingData, batch_size=self.batchSize)
            validationData = encoderModel.predict(validationData, batch_size=self.batchSize)
            testingData = encoderModel.predict(testingData, batch_size=self.batchSize)

            # Now regenerate noise for the new encoded space (important!)
            noisyTrainingData = trainingData + self.noiseFactor * np.random.normal(0.0, 1.0, trainingData.shape)
            noisyValidationData = validationData + self.noiseFactor * np.random.normal(0.0, 1.0, validationData.shape)



        # Writing all configurations to file 
        self._writeSDAConfigToFile(outputDirectory)

        # creating the full stacked sequential auto encoder model after training all layers
        finalModel = self._buildModelFromEncoders(encoders, dropoutAll = True)
        finalModel.compile(optimizer = self.optimizer, loss = self.lossFunction)
        finalModel.fit(
            self._batchGeneratorForTraining(
                    originalNoisyTrainingData, originalTrainingData,
                    self.batchSize,
                    shuffle=True
                ),
                epochs=self.numberOfEpochs,
                steps_per_epoch=int(np.ceil(originalNoisyTrainingData.shape[0] / self.batchSize)),
                callbacks=[earlyStoppingCallback],
                validation_data=self._batchGeneratorForTraining(
                    originalNoisyValidationData, originalValidationData,
                    self.batchSize,
                    shuffle=False
                ),
                validation_steps=int(np.ceil(originalNoisyValidationData.shape[0] / self.batchSize))
        )
        # Saving the file into the directory. 
        self._saveModel(finalModel, outputDir = outputDirectory, architectureFileName = f'enc_{self.layerType}_{self.activationType}_layers.png', modelJsonFileName = f'enc_{self.layerType}_{self.activationType}_layers.json', weightsFileName = 'enc_layers.weights.h5')
        
        
        return finalModel, trainingData, validationData, testingData, reconstructionMSE
    
    # This saves the trained model
    def _saveModel(self, model, outputDir, architectureFileName = 'model_arch.png', modelJsonFileName = 'model_arch.json', weightsFileName = 'model_weights.h5'):
        '''
        Saves a Keras model description and model weights
        @param model: a keras model
        @param outputDir: directory to save model architecture and weights to
        @param modelJsonFileName: file name for model architecture
        @param weightsFileName: filename for model weights
        '''
        print('saving the model')
        model.summary() # prints the model structure. 
        plot_model(model, to_file=outputDir+architectureFileName) # provides a visual diagram for the model. 
        
        # exporting the architecture to a json file
        jsonString = model.to_json()
        open(outputDir+modelJsonFileName, 'w').write(jsonString)
        
        # and saving its weights.
        model.save_weights(outputDir+weightsFileName, overwrite=True)

        keras_model_path = outputDir + f'final_sdae_{self.layerType}_{self.activationType}.keras'
        print(f"\nSaving full model to {keras_model_path}")
        model.save(keras_model_path)
        print("Model saved successfully in .keras format")

        
    # This stores the SDA hyperparameters to a file for future reference.
    def _writeSDAConfigToFile(self, outDir):
        """
        Write the configuration of the autoencoder to a file
        """
        print('writing SDA config')
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
    def _buildModelFromEncoders(self, encodingModels, dropoutAll=False, addDecoder=True):
        """
        Build either:
        (a) a stacked encoder model (for feature extraction), or
        (b) a full autoencoder (encoder + decoder) for end-to-end fine-tuning.

        @param encodingModels: list of pretrained encoder models (each mapping input→encoded)
        @param dropoutAll: whether to insert dropout layers between encoders
        @param addDecoder: if True, mirrors the encoder to build a full autoencoder
        @return: Keras Model (stacked encoder or autoencoder)
        """
        print("Building stacked model from encoders" + (" + decoder" if addDecoder else ""))

        model = Sequential()
        input_dim = encodingModels[0].input.shape[-1]
        model.add(Input(shape=(input_dim,)))

        # -------------------- ENCODER STACK --------------------
        encoder_shapes = []  # remember layer sizes for decoder
        for i, enc_model in enumerate(encodingModels):
            # find the actual Dense layer (with weights)
            enc_layer = next((l for l in reversed(enc_model.layers) if len(l.get_weights()) > 0), None)
            if enc_layer is None:
                raise ValueError(f"Encoder {i} has no trainable layer.")

            weights = enc_layer.get_weights()
            units = weights[0].shape[1]
            encoder_shapes.append(units)

            new_enc = Dense(
                units=units,
                activation=enc_layer.activation,
                use_bias=self.bias,
                name=f"stacked_enc_{i}_{self.layerType}_{self.activationType}"
            )
            new_enc.build((None, input_dim))
            new_enc.set_weights(weights)

            if i and dropoutAll:
                model.add(Dropout(self.dropoutPerLayer[i]))

            model.add(new_enc)
            input_dim = units

        if addDecoder:
            print("Adding symmetric decoder for fine-tuning.")
            # mirror encoder sizes in reverse (e.g., [256, 128] → decoder [128, 256, input_dim])
            for i, units in enumerate(reversed(encoder_shapes[:-1])):
                model.add(Dense(
                    units=units,
                    activation=self.decodingActivationPerLayer[-(i+1)],
                    name=f"stacked_dec_{i}_{self.layerType}_{self.activationType}"
                ))

            # final reconstruction layer (back to original input size)
            model.add(Dense(
                units=encodingModels[0].input.shape[-1],
                activation=self.decodingActivationPerLayer[0],
                name="final_reconstruction_layer"
            ))

        model.summary()
        dummy = np.zeros((1, 784), dtype=np.float32)
        model(dummy)
        return model

    def _buildModelFromEncoders2(self, encodingModels, dropoutAll=False):
        """
        Safely rebuilds a stacked encoder model from pretrained encoder models.
        Supports Dense and Conv1D encoders automatically.
        """
        print("Building stacked model from encoders")

        model = Sequential()
        input_dim = encodingModels[0].input.shape[-1]
        model.add(Input(shape=(input_dim,)))

        for i, enc_model in enumerate(encodingModels):
            # find last trainable layer (Dense or Conv1D)
            enc_layer = None
            for layer in reversed(enc_model.layers):
                if len(layer.get_weights()) > 0:
                    enc_layer = layer
                    break

            if enc_layer is None:
                raise ValueError(f"Encoder {i} has no trainable layer.")

            weights = enc_layer.get_weights()

            # ======= handle Dense layer =======
            if isinstance(enc_layer, Dense):
                units = weights[0].shape[1]
                new_enc = Dense(
                    units=units,
                    activation=enc_layer.activation,
                    use_bias=self.bias,
                    name=f"stacked_enc_{i}_{self.layerType}_{self.activationType}"
                )
                new_enc.build((None, input_dim))
                
                new_enc.set_weights(weights)
                if i and dropoutAll:
                    model.add(Dropout(self.dropoutPerLayer[i]))
                model.add(new_enc)
                input_dim = units

            # ======= handle Conv1D layer =======
            elif 'Conv1D' in enc_layer.__class__.__name__:
                filters = enc_layer.filters
                kernel_size = enc_layer.kernel_size[0]
                new_enc = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=enc_layer.strides[0],
                    padding=enc_layer.padding,
                    activation=enc_layer.activation,
                    use_bias=self.bias,
                    name=f"stacked_enc_{i}_{self.layerType}_{self.activationType}"
                )
                new_enc.build((None, input_dim, weights[0].shape[1]))
                new_enc.set_weights(weights)
                if i and dropoutAll:
                    model.add(Dropout(self.dropoutPerLayer[i]))
                model.add(new_enc)
                input_dim = filters  # update for next layer

            else:
                raise TypeError(f"Unsupported encoder layer type: {type(enc_layer)}")

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
            
        # Handle negative layer index (like -1)
        target_layer = model.layers[layerNumber]
        
        # Build a new model that outputs from this layer
        intermediate_model = Model(inputs=model.input, outputs=target_layer.output)
        
        # Use predict() to get the output
        return intermediate_model.predict(inputData, batch_size=self.batchSize)
    


    def _getReconstructionError(self, model, inputData, numberOfNeurons):
        """
        Return reconstruction squared error at individual nodes, averaged across all instances.
        @param model: trained model
        @param inputData: input data to reconstruct
        @param numberOfNeurons: number of model output nodes
        """
        # -1 is the last layer
        trainReconstruction = self._getIntermediateLayerOutput(model, inputData, layerNumber = -1, applyDropOut = 0, numberOfNeurons = numberOfNeurons, batchSize = self.batchSize) #train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of third layer: output layer
        
        # mean error
        reconMSE = np.mean(np.square(trainReconstruction - inputData), axis = 0)
        
        reconMSE = np.ravel(reconMSE)
        
        return reconMSE

    def _batchGeneratorForExtraction(self, X, batchSize):
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



    def _batchGeneratorForTraining(self, X, Y, batch_size, shuffle, seed = 1337):
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