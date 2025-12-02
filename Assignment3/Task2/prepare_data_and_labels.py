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

        # S7 beginnt bei Offset 7, Header-Länge ~12 Bytes:
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
def pcaps_feature_and_attack_label_extraction(input_path_attack_pcap,input_path_control_pcap,output_file_raw_features,output_file_re_features, output_file_labels_raw, output_file_labels_re, output_file_timestamps_raw, output_file_timestamps_re):

    pcap_files_control=list_files_by_filetype(input_path_control_pcap,"pcap")
    pcap_files_attack = list_files_by_filetype(input_path_attack_pcap, "pcap")
    #print(pcap_files_attack)

    full_byte_list_raw=[]
    full_label_list_raw = []
    full_timestamp_list_raw = []

    full_byte_list_re=[]
    full_label_list_re = []
    full_timestamp_list_re=[]

    #control files!
    for path in pcap_files_control:


        # extract n bytes of length M for file
        byte_lists_for_file_raw, timestamps_file_raw = read_pcap_bytes_raw(path, 100)
        if not byte_lists_for_file_raw:
            print(f"No packets in {path}, skipping.")
            continue
        # convert bytes to features
            #todo (skip for now)

        # list with labels (n times "None")
        label_list_for_file_raw = ["CONTROL"] * len(byte_lists_for_file_raw)

        full_byte_list_raw.extend(byte_lists_for_file_raw)
        full_label_list_raw.extend((label_list_for_file_raw))
        full_timestamp_list_raw.extend(timestamps_file_raw)

        #RE
        byte_lists_for_file_re, timestamps_file_re=read_pcap_bytes_re(path,100)
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

        byte_lists_for_file_raw, timestamps_file_raw = read_pcap_bytes_raw(path, 100)

        if not byte_lists_for_file_raw:
            print(f"No packets in {path}, skipping.")
            continue

        label_list_for_file_raw = [label_for_file] * len(byte_lists_for_file_raw)

        full_byte_list_raw.extend(byte_lists_for_file_raw)
        full_label_list_raw.extend((label_list_for_file_raw))
        full_timestamp_list_raw.extend(timestamps_file_raw)

        byte_lists_for_file_re,timestamps_file_re = read_pcap_bytes_re(path, 100)
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
    full_feature_array_raw=np.array(full_byte_list_raw, dtype=np.uint8)
    full_label_array_raw=np.array(full_label_list_raw)
    full_timestamp_array_raw=np.array(full_timestamp_list_raw)
    np.save(output_file_raw_features, full_feature_array_raw)
    np.save(output_file_labels_raw, full_label_array_raw)
    np.save(output_file_timestamps_raw, full_timestamp_array_raw)

    #save re array and labels
    full_feature_array_re = np.array(full_byte_list_re, dtype=np.uint8)
    full_label_array_re = np.array(full_label_list_re)
    full_timestamp_array_re=np.array(full_timestamp_list_re)

    np.save(output_file_re_features, full_feature_array_re)
    np.save(output_file_labels_re, full_label_array_re)
    np.save(output_file_timestamps_re, full_timestamp_array_re)

    return 0


