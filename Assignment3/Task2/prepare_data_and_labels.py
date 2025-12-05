from scapy.all import PcapReader, rdpcap, IP, IPv6, TCP, UDP, Ether
from scapy.error import Scapy_Exception

from Assignment3.Task2.constants import ATTACK_LABELS
from file_helper_t3 import *
import numpy as np


#input: pcap file
#output: Liste von Byte-Listen (jeweils Länge M) NACH RE-Logik
def read_pcap_bytes_re(pcap_file, M):
    """
    Extracts only valid S7Comm packets from TCP payload, efficient version:

    Annahmen:
    - TCP-Payload-Struktur: [TPKT (4B)] [COTP (3B)] [S7 (>=12B)]
    - TPKT beginnt bei Payload-Offset 0
    - S7-Protokoll beginnt bei Payload-Offset 7
    - S7-Header hat mind. 12 Bytes, danach kommen "physical readings"

    Filter:
    - TCP Port 102 (src oder dst)
    - Payload mindestens 7 + 12 Bytes lang
    - Optional: TPKT-Check: payload[0:2] == 0x03 0x00
    - S7-Protokoll-ID: payload[7] == 0x32
    - Wir extrahieren ab Offset 7+11 (= 18) den "physical" Teil
    """

    try:
        if os.path.getsize(pcap_file) == 0:
            print(f"Skipping empty pcap: {pcap_file}")
            return [], []
        pkts = rdpcap(str(pcap_file))
    except Scapy_Exception as e:
        print(f"Skipping unreadable pcap {pcap_file}: {e}")
        return [], []

    packet_arrays_file = []
    timestamps_file=[]
    for p in pkts:
        # Muss TCP mit Port 102 sein
        if not p.haslayer(TCP):
            continue
        tcp = p[TCP]
        if tcp.sport != 102 and tcp.dport != 102:
            continue

        # Nur TCP-Payload betrachten, nicht den ganzen Frame
        if not tcp.payload:
            continue
        payload = bytes(tcp.payload)

        # min TPKT(4) + COTP(3) + S7(12)
        MIN_LEN = 7 + 12
        if len(payload) < MIN_LEN:
            continue

        # tpkt check (optional)
        if payload[0] != 0x03 or payload[1] != 0x00:
            continue

        # Dein Wunsch: "an Stelle 8 der 0x32 Header" -> Index 7
        if payload[7] != 0x32:
            continue

        # S7 starts at offset 7, header length 12 Bytes:
        s7_start = 7
        s7_header_len = 12  #position of keyword
        phys_start = s7_start + s7_header_len

        if phys_start >= len(payload):
            continue

        physical_bytes = payload[phys_start:]
        if not physical_bytes:
            continue

        timestamp = p.time
        fixed = to_fixed_bytes(physical_bytes, M)
        packet_arrays_file.append(fixed)
        timestamps_file.append(timestamp)
    return packet_arrays_file, timestamps_file


#input: single packet in binary, length M
#output: array for packet with required length
def to_fixed_bytes(raw, M) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)    #convert bytes to numpy array
    # truncate if longer than M
    if arr.size >= M:
        return arr[:M]

    #pad if smaller than M
    out = np.zeros(M, dtype=np.uint8) #array of length M filled with zeros
    out[:arr.size] = arr   #replace first part with original array
    return out


#input: pcap file
#output: packet list for that file [[2,244,23,...],[33,213,112,..]], list with timestamps
def read_pcap_bytes_raw(pcap_file, M):
    try:
        # check if file is empty
        if os.path.getsize(pcap_file) == 0:
            print(f"Skipping empty pcap: {pcap_file}")
            return [], []

        pkts = rdpcap(str(pcap_file))
    except Scapy_Exception as e:
        print(f"Skipping unreadable pcap {pcap_file}: {e}")
        return [], []


    packet_arrays_file=[]
    timestamps_file=[]
    for p in pkts:
        timestamp = p.time
        raw=bytes(p)
        packet_arrays_file.append(to_fixed_bytes(raw, M))
        timestamps_file.append(timestamp)
    return packet_arrays_file, timestamps_file



#input - path to file (eg: /20161215203826.WaterTankOff_Flooding/20161215203826.WaterTankOff_Flooding.hmi.pcap')
#output - label that identifies attack type (WaterTankOff_Flooding)
def extract_attack_type(pcap_path: str):
    """
    input  - voller Pfad zur pcap-Datei
    output - Attack-Label exakt wie im Dateinamen (inkl. _Flooding) oder None
    """

    name = Path(pcap_path).name
    parts = name.split(".")

    # 20161215190548.WaterTankOnManu_Flooding.attacker.pcap
    # -> ['20161215190548', 'WaterTankOnManu_Flooding', 'attacker', 'pcap']
    # 20161216202830.master.pcap
    # -> ['20161216202830', 'master', 'pcap']

    if len(parts) < 3:
        # sowas wie 'foo.pcap' -> kein Attack-Label
        return None #skip in caller function

    candidate = parts[1]

    if candidate in ATTACK_LABELS:
        return candidate

    return None #skip in caller function



#used for QUT_S7Comm
#output: file with features for the complete byte length (raw), file with features for byte length starting from keyword (re), file with label for each index of the feature files
def pcaps_byte_and_metadata_extraction(input_path_attack_pcap,input_path_control_pcap,output_file_bytes_raw, output_file_bytes_re, output_file_labels_raw, output_file_labels_re, output_file_timestamps_raw, output_file_timestamps_re,M_raw,M_re):
    """

    Saves to file:
    -> list of bytes per pcap packet (for raw and re)
    -> list of all labels (for raw and re)
    -> list of all timestamps (raw and re)
    indices align between all raw files and all re files (not between raw and re files)
    """

    pcap_files_control=list_files_by_filetype(input_path_control_pcap,"pcap")
    pcap_files_attack = list_files_by_filetype(input_path_attack_pcap, "pcap")


    full_byte_list_raw=[]
    full_label_list_raw = []
    full_timestamp_list_raw = []

    full_byte_list_re=[]
    full_label_list_re = []
    full_timestamp_list_re=[]

    #control files!
    for path in pcap_files_control:


        # extract n bytes of length M for file
        byte_lists_for_file_raw, timestamps_file_raw = read_pcap_bytes_raw(path, M_raw)
        if not byte_lists_for_file_raw:
            print(f"No packets in {path}, skipping.")
            continue

        # list with labels (n times "None")
        label_list_for_file_raw = ["CONTROL"] * len(byte_lists_for_file_raw)

        full_byte_list_raw.extend(byte_lists_for_file_raw)
        full_label_list_raw.extend((label_list_for_file_raw))
        full_timestamp_list_raw.extend(timestamps_file_raw)

        #RE
        byte_lists_for_file_re, timestamps_file_re=read_pcap_bytes_re(path,M_re)
        if not byte_lists_for_file_re:
            print(f"No packets in {path}, skipping.")
            continue
        label_list_for_file_re = ["CONTROL"] * len(byte_lists_for_file_re)
        full_byte_list_re.extend(byte_lists_for_file_re)
        full_label_list_re.extend((label_list_for_file_re))
        full_timestamp_list_re.extend(timestamps_file_re)

        print(f"File {path} done.")

    print("\n\nControl dataset done\n\n")

    #attack files!
    for path in pcap_files_attack:

        label_for_file = extract_attack_type(path)

        #skip pcap files that are not in one of the "attack label diretories"
        # (-> eg huge pcap files in the parent directory that summarize them)
        if label_for_file is None:
            continue

        byte_lists_for_file_raw, timestamps_file_raw = read_pcap_bytes_raw(path, M_raw)

        if not byte_lists_for_file_raw:
            print(f"No packets in {path}, skipping.")
            continue

        label_list_for_file_raw = [label_for_file] * len(byte_lists_for_file_raw)

        full_byte_list_raw.extend(byte_lists_for_file_raw)
        full_label_list_raw.extend((label_list_for_file_raw))
        full_timestamp_list_raw.extend(timestamps_file_raw)

        byte_lists_for_file_re,timestamps_file_re = read_pcap_bytes_re(path, M_re)
        if not byte_lists_for_file_re:
            print(f"No packets in {path}, skipping.")
            continue
        label_list_for_file_re = [label_for_file] * len(byte_lists_for_file_re)
        full_byte_list_re.extend(byte_lists_for_file_re)
        full_label_list_re.extend((label_list_for_file_re))
        full_timestamp_list_re.extend((timestamps_file_re))
        print(f"File {path} done.")

    print("\n\nAttack dataset done\n\n")

    #save raw array and labels
    full_label_array_raw=np.array(full_label_list_raw)
    full_timestamp_array_raw=np.array(full_timestamp_list_raw)
    np.save(output_file_bytes_raw, full_byte_list_raw)
    np.save(output_file_labels_raw, full_label_array_raw)
    np.save(output_file_timestamps_raw, full_timestamp_array_raw)

    #save re array and labels
    full_label_array_re = np.array(full_label_list_re)
    full_timestamp_array_re=np.array(full_timestamp_list_re)
    np.save(output_file_bytes_re, full_byte_list_re)

    np.save(output_file_labels_re, full_label_array_re)
    np.save(output_file_timestamps_re, full_timestamp_array_re)

    return 0


#input: pcap file
#output: size of longest packet considering all packets
def size_longest_packet_from_pcap_raw(pcap_file):
    try:
        # check if file is empty
        if os.path.getsize(pcap_file) == 0:
            print(f"Skipping empty pcap: {pcap_file}")
            return 0

        pkts = rdpcap(str(pcap_file))
    except Scapy_Exception as e:
        print(f"Skipping unreadable pcap {pcap_file}: {e}")
        return 0

    max_length=0
    for p in pkts:
        raw_length=len(bytes(p))
        if raw_length>max_length:
            max_length=raw_length
    return max_length


# input: pcap file
# output: size of the longest S7Comm payload (integer) after RE filters
def size_longest_packet_from_pcap_re(pcap_file):
    """
    Applies S7Comm RE filters (Port 102, S7 protocol checks, etc.)
    and returns the length of the longest 'physical_bytes' segment found.
    """

    # Remove the unused parameter M and the dependency on to_fixed_bytes

    # 1. Initialize the maximum length
    max_length = 0

    try:
        if os.path.getsize(pcap_file) == 0:
            print(f"Skipping empty pcap: {pcap_file}")
            # Returns 0 for empty file
            return max_length

        pkts = rdpcap(str(pcap_file))
    except Scapy_Exception as e:
        print(f"Skipping unreadable pcap {pcap_file}: {e}")
        # Returns 0 for unreadable file
        return max_length

    # Remove unused lists from the original function
    # packet_arrays_file = []
    # timestamps_file=[]

    for p in pkts:
        # Muss TCP mit Port 102 sein
        if not p.haslayer(TCP):
            continue
        tcp = p[TCP]
        if tcp.sport != 102 and tcp.dport != 102:
            continue

        # Nur TCP-Payload betrachten
        if not tcp.payload:
            continue
        payload = bytes(tcp.payload)

        # min TPKT(4) + COTP(3) + S7(12)
        MIN_LEN = 7 + 12
        if len(payload) < MIN_LEN:
            continue

        # tpkt check (optional)
        if payload[0] != 0x03 or payload[1] != 0x00:
            continue

        # S7-Protokoll-ID: payload[7] == 0x32
        if payload[7] != 0x32:
            continue

        # S7 beginnt bei Offset 7, Header-Länge ~12 Bytes:
        s7_start = 7
        s7_header_len = 12
        phys_start = s7_start + s7_header_len

        if phys_start >= len(payload):
            continue

        # This is the actual feature data we care about (the extracted payload)
        physical_bytes = payload[phys_start:]

        if not physical_bytes:
            continue

        # 2. Calculate the length of the extracted bytes
        current_length = len(physical_bytes)

        # 3. Update max_length if the current extracted payload is longer
        if current_length > max_length:
            max_length = current_length
    return max_length


def find_M(input_path_control_pcap, input_path_attack_pcap):
    """length of longest packets in pcap, necessary for Autoencoder training and testing
    raw-> for all packets
    re -> for all s7comm packets, only considering the part with physical readings"""
    pcap_files_control = list_files_by_filetype(input_path_control_pcap, "pcap")
    pcap_files_attack = list_files_by_filetype(input_path_attack_pcap, "pcap")

    # Initialize maximum lengths
    max_packet_length_raw = 0
    max_packet_length_re = 0

    # --- Control Files ---
    print("Starting control dataset analysis...")
    for path in pcap_files_control:
        max_packet_length_file_raw=size_longest_packet_from_pcap_raw(path)
        if max_packet_length_raw < max_packet_length_file_raw:
            max_packet_length_raw=max_packet_length_file_raw

        # RE
        max_packet_length_file_re = size_longest_packet_from_pcap_re(path)
        if max_packet_length_re < max_packet_length_file_re:
            max_packet_length_re=max_packet_length_file_re


        print(f"File {path} done. Current Max Raw: {max_packet_length_raw}, Current Max RE: {max_packet_length_re}")

    print("\n\nControl dataset done\n\n")

    # --- Attack Files ---
    print("Starting attack dataset analysis...")
    for path in pcap_files_attack:

        max_packet_length_file_raw = size_longest_packet_from_pcap_raw(path)
        if max_packet_length_raw < max_packet_length_file_raw:
            max_packet_length_raw = max_packet_length_file_raw

        # RE
        max_packet_length_file_re = size_longest_packet_from_pcap_re(path)
        if max_packet_length_re < max_packet_length_file_re:
            max_packet_length_re = max_packet_length_file_re

        print(f"File {path} done. Current Max Raw: {max_packet_length_raw}, Current Max RE: {max_packet_length_re}")

    print("\n\nAttack dataset done\n\n")

    print(f"Final Max packet length (raw): {max_packet_length_raw}")
    print(f"Final Max packet length (re): {max_packet_length_re}")

    return max_packet_length_raw, max_packet_length_re
