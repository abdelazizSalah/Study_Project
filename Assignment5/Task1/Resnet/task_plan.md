# Assignment 5 (Task1 - ResNet implementation)
- In this task we are asked to implement ResNet for anomaly detection
- Main goal is to build ResNet-18 deeplearning classifier that detects Normal vs Attack
- It should work on:
  - Raw bytes
  - RE with p:
    - p = 5
    - p = 10
    - p = 15
- Each input should again fit M bytes, so it should be either truncated or padded. 

## Task Plan

### Phase 0: Design decision
- Framework: PyTorch
- Convolution type: 1D ResNet
  - Input shape: (Batch, channels = 1, length= M)
  - Reason: bytes are seuence, 1D conv is natrual and stable.
- Padding/trimming rule: we will take the first M bytes, and pad shorted packets with 0
- Cross-entropy:
  - Requirement says categorical cross-entropy, so I will implement it using nn.CrossEntropyLoss() and output logits of size 2 (normal/ attack). 

### Phase 1: Data pipeline
1. Command-line interface (exactly similar to Sheet4)
   - -M number of bytes per sample
   - --use_stats: whether to include the stats or not
   - --p: use physical reading count for RE mode
2. Preprocessing (exactly similar to Sheet4)
   - Build extract_bytes function which return np.array of size M. 
   - Then normalize the bytes:
     - x = bytes / 255
3. Dataset + DataLoader
   - Dataset returns:  
     - x_bytes: shape (1, M)
     - x_stats: shape (5)      
     - y: 0/1
4. Splits
   - We should use Anna's previously created folds.
   - ensure that we can run train/val/testing per fold.

### Phase 2: Implementation of ResNet-18 backbone (1D)
1. Residual block (basicBlock 1D)
   1. 2 convolution layers per block:
      - Conv1D -> Batch Normalization (BN) -> ReLU -> conv1D -> BN     
   2. Skip connection:
      1. identity if shapes match
      2. projection (1x1 Conv1d) if channels/stride changes
2. ResNet-18 stages
   - it should consist of 4 stages, each stage has 2 residual blocks
   - Standard channel progression example: 
     1. Stage1: 64
     2. Stage2: 128
     3. Stage3: 256
     4. Stage4: 512
   - Total "18 Layers" requirement: 
     - follow the ResNet-18 pattern (counts alogn when we consider convs)
3. Pooling
   - We can use AdaptiveAvgPool1d(1) at the end
     - it gives fixed-size embedding regardless of M
4. 2 Fully connected layers
   1. FC1 (embeddings -> hidden) + ReLU + DropOut
   2. FC2 (hidden -> 2 logits)
5. Sanity checks
   - we can run on one batch to check:
     - shapes
     - logits shape
     - loss runs
### Phase 3: Implementation of 5 statistical features
1. Required 3 features:
   - Total byte length before trim/pad (original length)
   - Count of the most frequent byte
   - Count of the second most frequent byte           
2. Two extra features
   - Byte entropy
     - measures randomness/structure; attacks may change distribution
   - Mean absolute difference between consecutive bytes
     - captures burstiness/structure changes
3. Integration with ResNet
   - ResNet produces embedding h (e.g. 512-dim)   
   - Concatenate: h_cat = [h, stats] -> shape = 512 + 5
   - Then we should feed them into the FC layers

### Phase 4: Training Loop
1. Forward pass:
   - logits = model(x_bytes, x_stats)
2. Loss
   - Loss = CrossEntropyLoss(logits,y)
3. Optimization
   - Adam
4. Logging
   - Train loss
   - Val loss
   - Percision/Recall/F1 on val each epoch (important for selecting best model) 

### Phase 5: Evaluation + metrics (per fold)
- For each fold:
  - Evaluate on testset: 
    - Percision
    - Recall
    - F1
  - save:
    - predictions
    - confusion matrix
    - metrics JSON/CSV
### Phase 6: Hyperparameter tuning 
- perform fine-tuning on the parameters:
  - learning_rate
  - batch_size
  - dropout
  - weight_decay
  - epochs

### Phase 7: add it to the toolbox
- integrate it with our toolbox. 