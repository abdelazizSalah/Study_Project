import numpy as np

#expects vector from dataset tupel as input!
def classify_single_packet(trained_model, threshold, x):
    # If this comes from a Dataset that yields (x, y), take x

    # Ensure 1D NumPy vector of floats
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D packet vector (shape (M,)), got {x.shape}")

    # Add batch dimension for Keras: (1, M)
    x_in = x.reshape(1, -1)

    # Reconstruction from autoencoder
    x_rec = trained_model.predict(x_in, verbose=1)  # shape (1, M)

    # Per-sample MSE = mean over features
    mse = float(np.mean(np.square(x_in - x_rec)))

    is_attack = mse > threshold
    return is_attack, mse



#assumes that datasets are already present in dataset_attack.npy and dataset_normal.py!!
#if necessary create them in main using prepare_data.store_dataset_in_file
def test_classifier(trainedModel, y):
    #load dataset from file
    attack_data = np.load("dataset_attack.npy")
    normal_data = np.load("dataset_normal.npy")
    #ds_all = tf.data.Dataset.from_tensor_slices(data)
    #ds_all_batched = ds_all.batch(128)
    #mean_error=getMSEDataset(trainedModel, ds_all_batched)
    #print(mean_error)

    subset_attack = attack_data[:1000]  # first 1000 packets
    subset_normal = normal_data[:1000]

    print(f"Threshold for classification: {y}")

    #no mixing required
    tp=0
    fn=0
    for sample in subset_attack:
        is_attack, mse = classify_single_packet(trainedModel, y, sample)
        if(not is_attack):
            fn=fn+1
        else:
            tp=tp+1
    print(f"Attack Packets: True Positives:{tp} | False Negatives:{fn}\n")
    #threshold 0.0001: True positives:961 | False Negatives:39

    tn = 0
    fp = 0
    for sample in subset_normal:
        is_attack, mse = classify_single_packet(trainedModel, y, sample)
        if (not is_attack):
            tn = tn + 1
        else:
            fp = fp + 1
    print(f"Normal Packets: True Negatives:{tn} | False Positives:{fp}\n")
    # threshold 0.0001 - Normal Packets: True Negatives:864 | False Positives:136
    total = tp + fn + tn + fp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")

    #threshold 0.0001
    #Accuracy: 0.9125
    #Precision: 0.8760255241567912
    #Recall: 0.961
    return


#used to compute MSE for normal packets (to choose a good threshold and verify results)
#expects dataset in batches
def getMSEDataset(trainedModel, ds_all):
    X_rec = trainedModel.predict(ds_all, verbose=1)

    #model predicts average error for one vector
    # If ds yields only features:
    X_all = np.concatenate([x.numpy() for x in ds_all], axis=0)

    # Compute per-sample errors
    errors = np.mean(np.square(X_all - X_rec), axis=1)
    mean_error = np.mean(errors)

    return mean_error