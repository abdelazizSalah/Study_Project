- for task2.d I ran the TSNE on 10% of the normal data, to conduct hyper parameters tuning, and this was on 06.11 and it till now did not finish, and also for the e, I ran a process with the full CPU cores, and it did not finish, and this makes sense because it is 300M C 2 which is almost ( 300M^2 / 2 ). 
  - For pairs, try to compute as much as possible till the submission day.
  - and for the hyper parameter tuning, try to check which range is the best, and then based on it, decide what range to use to reduce the search space. 
- for task2.d t-SNE is used when we need to reduce from high-dimensions to low dimensions, but in our case, we already have only 3 dimensions, which is already low dimensions. So the question is do you want to perform it only on the selected 3 features, or we need to perform it on the whole features before prepration? 
  - The number of dimensions is the number of packets within each flow, not the 3 featuers we have selected. 
- In task 3.b, you say two different types of layers, you mean here Dense and Sparse for example, so the whole SDA contains for example Dense layers and each with different activation function, then another 3 with Sparse layer for example.
    - Dense and sparse.  
- What are the hyperparameters you mean in task 3.b, are they the weights of the connections between the nodes, or this ?
    - Also consider the parameters defined for within the layers themselves. 
  -  the number of layers, the size of each layer, the activation function (e.g., sigmoid, tanh), the type and amount of noise added, the optimizer and learning rate, and the loss function (e.g., mean squared error).
-  The SDA contains a parameter called noising and corruption, and this usually can work for images for example, to add some sort of distortion to the data, but how can I apply it in our case, because it is very strange? 
   -  My suggestion will be to add that noise to the input data, like summing some of them to the numerical values like packet size and so on, and changing the direction for some of them also, or is this not so important? 
- In task 3, it is mentioned that the SDA should take input m*n and return as output only n, but I think you meant n for each packet, because the functionality of the autoencoder, is that it takes input, and then reconstruct the input once again from the decoder part, or you meant here that you want to extract the output from the encoder last layer which is the latent space?
    -in other words, do you mean that the bottelneck layer should have size M, or the output of the decoder is of size M, because if the bottelneck layer will be of size M, then the autoencoder will not reduce the number of neurons on each level, and it will stay fixed always.
      - Yes, for each packet it should extract n features. 
- in task 3.d the o here makes more sense to be the decoded vector, because the input here is not compatible with the encoded vector, they are different in shape. and also this is not the reconstruction error. 
  - yes it should be the decoder not encoder.


- Add noise to the input. 