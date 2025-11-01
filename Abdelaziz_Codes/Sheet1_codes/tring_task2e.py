from scapy.all import rdpcap
import os

#list all files of certain filetype from directory and it's subdirectories
def list_files_by_filetype(root_path, filetype):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith("."+filetype):
                full_path = os.path.join(dirpath, filename)
                pcap_files.append(full_path)
    return pcap_files

def read_pcap_as_byte_sequences(pcap_path):
    packets = rdpcap(pcap_path)              # Load all packets
    byte_sequences = [bytes(pkt) for pkt in packets]  # Convert each packet to raw bytes
    return byte_sequences

# Example usage
if __name__ == "__main__":
    # pcap_file = "E:/GitHub/Study_Project/DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7Comm/control/hmi.pcap.csv"
    pcap_files = list_files_by_filetype('E:/GitHub/Study_Project/DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set', "pcap")
    print(pcap_files)
    pcap_file = pcap_files[0]
    byte_packets = read_pcap_as_byte_sequences(pcap_file)
    print(f"Total packets: {len(byte_packets)}")
    print("First packet bytes:", list(byte_packets[0])[:50])  # show first 50 bytes
