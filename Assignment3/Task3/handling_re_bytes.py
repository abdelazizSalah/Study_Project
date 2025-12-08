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
        concatenated_packet = np.concatenate(data[i:i + p])
        concatenated_data.append(concatenated_packet)

    # handle the case where len(data) is not a multiple of p
    if remainder != 0:
        concatenated_packet = np.concatenate(data[-remainder:])
        concatenated_data.append(concatenated_packet)  # to be repeated with remainder number of packets later.
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
def save_npy_file(data, file_path, p, remainder):
    """
        - repeat each data item p times first, and last item remainder times if remainder != 0, else p times.
    """
    stopIdx = 0
    if remainder != 0:
        stopIdx = len(data) - 2
    else:
        stopIdx = len(data)
    print('Saving processed data to .npy file...')
    expanded_data = []
    for i in range(len(data) - 1 - remainder):
        for _ in range(p):
            expanded_data.append(data[i])
    # handle the last item
    last_repeats = remainder if remainder != 0 else p
    for _ in range(last_repeats):
        expanded_data.append(data[-1])

    expanded_data = np.array(expanded_data)
    np.save(file_path, expanded_data)
    print(f"Processed data saved to {file_path}")
    print(f'len(expanded_data): {len(expanded_data)}')


def process_npy_file(input_file_path, output_file_path, p):
    """Process the .npy file as per the defined steps."""
    # Step 1: Load the .npy file
    data = load_npy_file(input_file_path)
    print(f'Original data length: {len(data)}')
    # Step 2: Exclude appended zeros from right
    data_no_zeros = exclude_appended_zeros(data)

    # Step 3: Concatenate every p packets' physical reading together
    concatenated_data, remainder = concatenate_packets(data_no_zeros, p)

    # Step 4: Determine the maximum length of the concatenated packets
    max_length = determine_max_length(concatenated_data)

    # Step 5: Append zeros to the right of the concatenated packets to make them of equal length
    padded_data = pad_packets(concatenated_data, max_length)

    # Step 6: Write the processed data to a new .npy file
    save_npy_file(padded_data, output_file_path, p, remainder)
    print(f'Processing complete. Output saved to {output_file_path}')
    return padded_data


if __name__ == "__main__":
    input_file_path = "re_bytes.npy"
    output_file_path = "processed_packets.npy"
    p = 5  # Number of packets to concatenate
    process_npy_file(input_file_path, output_file_path, p)