import numpy as np

for p in [5, 10, 15]:
    input_data_pat = f'processed_packets_{p}_dedup.npy'
    print(f"Loading deduplicated data from: {input_data_pat}")
    dedup_data =  np.load(input_data_pat)
    print(f"Loaded {len(dedup_data)} samples for processed_packets_{p}.npy\n")

    input_labels_path = f're_labels_{p}_dedup_binary_attack1_control0.npy'
    print(f"Loading deduplicated labels from: {input_labels_path}")
    dedup_labels = np.load(input_labels_path)
    print(f"Loaded {len(dedup_labels)} labels for processed_packets_{p}.npy\n")

    print(f'first 10 samples of deduplicated data for processed_packets_{p}.npy: with their labels')
    for i in range(10):
        print(f"Sample {i}: Data: {dedup_data[i]}, Label: {dedup_labels[i]}")
        print(f"Sample {-i}: Data: {dedup_data[-i]}, Label: {dedup_labels[-i]}")
    # remove padding zeros from the trailing part of each sample until the first non-zero value from the end
    print('\nRemoving trailing zeros from each sample...\n')
    dedup_data = [
    sample[:np.nonzero(sample)[0][-1] + 1] if np.any(sample) else sample
    for sample in dedup_data
    ]

    for i in range(10):
        print(f"Sample {i}: Data: {dedup_data[i]}, Label: {dedup_labels[i]}")
        print(f"Sample {-i}: Data: {dedup_data[-i]}, Label: {dedup_labels[-i]}")
    
    print('removed trailing zeros from each sample.\n')

    # separate normal and attack samples
    normal_samples = [s for s, l in zip(dedup_data, dedup_labels) if l == 0]
    attack_samples = [s for s, l in zip(dedup_data, dedup_labels) if l == 1]


    print(f"\nprocessed_packets_{p}.npy - Normal samples: {len(normal_samples)}, Attack samples: {len(attack_samples)}\n")
    print('----------------------------------------\n')

    # save normal and attack samples to separate files
    normal_data_output_path = f'final_data/final_processed_packets_{p}_dedup_normal_data.npy'
    normal_indices_output_path = f'final_data/final_processed_packets_{p}_dedup_normal_indices.npy'
    np.save(normal_data_output_path, np.array(normal_samples, dtype=object))
    np.save(normal_indices_output_path, np.where(dedup_labels == 0)[0])
    print(f"Saved normal samples to: {normal_data_output_path}")

    attack_data_output_path = f'final_data/final_processed_packets_{p}_dedup_attack_data.npy'
    attack_indices_output_path = f'final_data/final_processed_packets_{p}_dedup_attack_indices.npy'
    np.save(attack_data_output_path, np.array(attack_samples, dtype=object))
    np.save(attack_indices_output_path, np.where(dedup_labels == 1)[0])

    print(f"Saved attack samples to: {attack_data_output_path}\n")

