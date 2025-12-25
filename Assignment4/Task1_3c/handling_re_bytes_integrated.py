'''
@ Author: Abdelaziz Neamatallah
@ Date: 05.12.25
@ Description:
    handling the reverse engineering of bytes - by concatenating every p packets' physical reading together
Main logic:
    1. read .npy file
    2. exclute the appended zeros from right.
    3. concatenate every p packets' physical reading together.
    4. determine the maximum length of the concatenated packets.
    5. append zeros to the right of the concatenated packets to make them of equal length.
    6. write the processed data to a new .npy file.

'''
from pathlib import Path

# 1.read .npy file
import numpy as np
def load_npy_file(file_path):
    """Load a .npy file and return its contents."""
    print('Loading .npy file...')
    data = np.load(file_path, allow_pickle=True)
    return data

#  2. exclute the appended zeros from right.
def exclude_appended_zeros(data):
    """Exclude appended zero bytes from the right of each packet."""
    print('Excluding appended zeros from right...')
    # print (f'first 5 packets before exclusion: {data[:5]}')
    processed_data = []
    for packet in data:
        # Find the index of the last non-zero byte
        last_non_zero_index = np.where(packet != 0)[0]
        if len(last_non_zero_index) > 0:
            last_index = last_non_zero_index[-1] + 1  # +1 to include the last non-zero byte
            processed_data.append(packet[:last_index])
        else:
            print("Warning: Packet contains only zeros.")
            processed_data.append(np.array([]))  # If all bytes are zero, return an empty array
    # print (f'first 5 packets after exclusion: {processed_data[:5]}')
    return processed_data

# 3.concatenate every p packets' physical reading together.
def concatenate_packets(data, p):
    """Concatenate every p packets' physical reading together."""
    # check the remainder of len(data) divided by p
    print('Concatenating packets...')
    remainder = len(data) % p
    print(f"Remainder of len(data) divided by p: {remainder}")
    concatenated_data = []
    for i in range(0, len(data), p):
        concatenated_packet = np.concatenate(data[i:i+p])
        concatenated_data.append(concatenated_packet)

    # handle the case where len(data) is not a multiple of p
    if remainder != 0:
        concatenated_packet = np.concatenate(data[-remainder:])
        concatenated_data.append(concatenated_packet) # to be repeated with remainder number of packets later.
    return concatenated_data, remainder

# 4.determine the maximum length of the concatenated packets.
def determine_max_length(concatenated_data):
    """Determine the maximum length of the concatenated packets."""
    print('Determining maximum length of concatenated packets...')
    max_length = max(len(packet) for packet in concatenated_data)
    return max_length
# 5.append zeros to the right of the concatenated packets to make them of equal length.
def pad_packets(concatenated_data, max_length):
    """Append zeros to the right of the concatenated packets to make them of equal length."""
    print('Padding packets to equal length...')
    padded_data = []
    for packet in concatenated_data:
        padded_packet = np.pad(packet, (0, max_length - len(packet)), 'constant')
        padded_data.append(padded_packet)
    return np.array(padded_data)
# 6.write the processed data to a new .npy file.
def save_npy_file(data, file_path, p,  remainder):
    """
        - repeat each data item p times first, and last item remainder times if remainder != 0, else p times.
    """
    stopIdx=0
    if remainder !=0:
        stopIdx = len(data)-2
    else:
        stopIdx= len(data)
    print('Saving processed data to .npy file...')
    expanded_data = []
    for i in range(len(data)-1 - remainder):
        for _ in range(p):
            expanded_data.append(data[i])
    # handle the last item
    last_repeats = remainder if remainder !=0 else p
    for _ in range(last_repeats):
        expanded_data.append(data[-1])

    expanded_data = np.array(expanded_data)
    np.save(file_path, expanded_data)
    print(f"Processed data saved to {file_path}")
    print(f'len(expanded_data): {len(expanded_data)}')
    return


def pad_or_truncate_packets(packets, target_length):
    processed = []
    for pkt in packets:
        if len(pkt) > target_length:
            pkt = pkt[:target_length]  # truncate
        elif len(pkt) < target_length:
            pkt = np.pad(pkt, (0, target_length - len(pkt)), mode='constant')
        processed.append(pkt)
    return np.array(processed)



def process_npy_file(input_source, output_file_path, p):
    """Process the .npy file as per the defined steps."""

    # 🔹 NEU: entweder direkt ein Array verwenden ODER Datei laden
    if isinstance(input_source, np.ndarray):
        data = input_source
    else:
        data = load_npy_file(input_source)

    print(f'Original data length: {len(data)}')

    # Step 2: Exclude appended zeros from right
    data_no_zeros = exclude_appended_zeros(data)

    # Step 3: Concatenate every p packets' physical reading together
    concatenated_data, remainder = concatenate_packets(data_no_zeros, p)

    # Step 4: Determine the maximum length of the concatenated packets
    max_length = 386    # sadsagsafvfzqfevfdsldhvodsvbdsi

    # Step 5: Pad or truncate to exactly 386
    padded_data = pad_or_truncate_packets(concatenated_data, max_length)

    # Step 6: Write the processed data to a new .npy file (optional)
    if output_file_path is not None:
        save_npy_file(padded_data, output_file_path, p, remainder)
        print(f'Processing complete. Output saved to {output_file_path}')
    else:
        print('Processing complete. No file saved (output_file_path=None).')

    return padded_data

def create_preprocessed_re_files():
    p=[5,10,15]
    input_file_path = "datasets/re_bytes.npy"

    for i in p:
        output_file_path = f"datasets/re_bytes_{i}.npy"
        process_npy_file(input_file_path, output_file_path, i)
    return


def create_preprocessed_re15():
    input_file_path = "datasets/re_bytes.npy"


    output_file_path = f"datasets/re_bytes_15.npy"
    process_npy_file(input_file_path, output_file_path, 15)
    return




###################remove duplicates from preprocessed files-----------------------------------------------------
import numpy as np
from pathlib import Path

def get_keep_indices_from_fold0(feature_dir: str, model_prefix: str) -> np.ndarray:
    """
    Load fold 0 features and compute indices to keep so that all duplicate
    rows are removed (only the first occurrence of each row is kept).
    """
    feature_dir = Path(feature_dir)
    fold0_path = feature_dir / f"{model_prefix}_features_fold0.npy"

    if not fold0_path.exists():
        raise FileNotFoundError(f"Feature file not found: {fold0_path}")

    feats0 = np.load(fold0_path)

    # unique over rows; index gives first position of each unique row
    _, unique_indices = np.unique(feats0, axis=0, return_index=True)

    # sort so we preserve original order
    keep_indices = np.sort(unique_indices)

    print(f"Fold 0: total samples = {len(feats0)}, "
          f"after dedup = {len(keep_indices)} "
          f"(removed {len(feats0) - len(keep_indices)} duplicates)")
    return keep_indices


def get_keep_indices_from_fold0_ae(feature_file) -> np.ndarray:
    """
    Load fold 0 features and compute indices to keep so that all duplicate
    rows are removed (only the first occurrence of each row is kept).
    """
    fold0_path = Path(feature_file)

    if not fold0_path.exists():
        raise FileNotFoundError(f"Feature file not found: {fold0_path}")

    feats0 = np.load(fold0_path)

    # unique over rows; index gives first position of each unique row
    _, unique_indices = np.unique(feats0, axis=0, return_index=True)

    # sort so we preserve original order
    keep_indices = np.sort(unique_indices)

    print(f"Fold 0: total samples = {len(feats0)}, "
          f"after dedup = {len(keep_indices)} "
          f"(removed {len(feats0) - len(keep_indices)} duplicates)")
    return keep_indices