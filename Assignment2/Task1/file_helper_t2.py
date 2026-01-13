import os
import pandas as pd
from scapy.all import PcapReader, rdpcap, IP, IPv6, TCP, UDP, Ether
from scapy.layers.sctp import SCTP

import numpy as np




def save_alignment_and_candidates_npz(filepath, aligned_messages, keyword_candidates):
    np.savez_compressed(
        filepath,
        aligned=np.array(aligned_messages, dtype=object),
        candidates=np.array(keyword_candidates, dtype=object)
    )
    print("Saved compressed:", filepath)


def load_alignment_and_candidates_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["aligned"].tolist(), data["candidates"].tolist()


def save_df_as_parquet(df, path):
    df.to_parquet(path, compression="zstd", index=False)
    return 0


def read_df_from_parquet(path):
    df = pd.read_parquet(path, engine="pyarrow")
    return df


def save_df_to_csv(df,output_path, mode='w', header =True):
    df.to_csv(output_path, mode=mode, header=header, index=False)
    return 0


def read_df_from_csv(path):
    df=pd.read_csv(path, on_bad_lines="skip", engine="pyarrow") #pyarrow can read dfs faster
    return df


#list all files of certain filetype from directory and it's subdirectories
def list_files_by_filetype(root_path, filetype):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith("."+filetype):
                full_path = os.path.join(dirpath, filename)
                pcap_files.append(full_path)
    return pcap_files




#used for QUT_S7Comm
#one large file for attack dataset and control dataset
def create_large_csv_file_from_pcaps(input_path_attack_pcap,input_path_control_pcap,output_csv_file):

    pcap_files_attack=list_files_by_filetype(input_path_attack_pcap,"pcap")
    pcap_files_control=list_files_by_filetype(input_path_control_pcap,"pcap")

    first_control_file=1
    for path in pcap_files_control:

        filename = os.path.basename(path)
        try:
            write_header = first_control_file

            if first_control_file:
                file_mode = 'w'
                first_control_file=0
            else: file_mode='a'

            df_pcap=pcap_extract_values(path, 0)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=write_header)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nControl dataset done\n\n")

    first_attack_file = 1
    for path in pcap_files_attack:
        filename = os.path.basename(path)

        try:
            file_mode='a'
            df_pcap = pcap_extract_values(path, 1)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=0)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")


    return 0


# known_ports.py
KNOWN_PORTS = {
    102:    "s7comm",
    502:    "modbus",
    20000:  "dnp3",
    44818:  "ethernetip",
    34964:  "profinet",
    123: "NTP",
    137: "NBNS",
    138: "NBDS",
    546: "DHCPv6 Client",
    547: "DHCPv6 Server",
    5353: "mDNS",
    5355: "llmnr"

}


#
#input: port numbers
#output: application layer protocol
def get_app_proto(srcport,dstport):
    #assume that either the dstport or the srcport is well known
    ports=[srcport,dstport]

    for p in ports:
        if p in KNOWN_PORTS:
            return KNOWN_PORTS[p]

    return(f"unknown") # if there is no application layer or port not found


#whole application layer (including header) is counted as payload!
def header_and_payload_len(pkt):

    try:
        frame_len = len(pkt.original)  # on-wire bytes
    except Exception:
        frame_len = len(pkt)  # fallback

    app_len = 0

    if TCP in pkt:
        app_len = len(pkt[TCP].payload)  # S7comm (COTP/TPKT+S7) lives here as Raw
    elif UDP in pkt:
        app_len = len(pkt[UDP].payload)  # e.g., NTP, DNS, etc.
    elif SCTP in pkt:
        app_len = len(pkt[SCTP].payload)
    else:
        raise ValueError(f"ERROR in header_and_payload_len (unsupported packet type {pkt.summary()})")
        #raise error because packets without application layer shouldnt be used with this function

    header_len = max(frame_len - app_len, 0)
    return header_len, app_len


#creates pair_id for host pair (two hosts that communicate with each other)
def host_pair_id(src,dst):
    # replace with empty string if None
    a = src or ""
    b = dst or ""
    return "__".join(sorted([a, b]))




#input: pcap file
#output: dataframe with extracted value per packet
def pcap_extract_values(pcap_path, attack):

    records = []
    last_ts_by_pair = {}
    last_ts_by_proto_pair = {}
    pkt_index=0

    first_ts=None

    filename=os.path.basename(pcap_path)
    with (PcapReader(pcap_path) as pcap):

        for pkt in pcap:

            #calculate relative timestamp starting from the first packet of each pcap file
            try:
                ts = float(pkt.time)
                if first_ts is None:
                    first_ts = ts
                ts_rel = ts - first_ts  # relative seconds since first packet
                ts_rel=ts_rel
            except Exception:
                # if timestamp missing, set 0.0 (row still recorded)
                ts = 0.0

            try:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
            except Exception:
                src_ip = ""
                dst_ip = ""

            if Ether in pkt:
                src_mac = pkt[Ether].src
                dst_mac = pkt[Ether].dst
            else:
                src_mac =""
                dst_mac=""

            #extract port numbers
            try:
                if TCP in pkt:
                    src_port = int(pkt[TCP].sport)
                    dst_port = int(pkt[TCP].dport)
                    l4_proto="TCP"
                elif UDP in pkt:
                    src_port = int(pkt[UDP].sport)
                    dst_port = int(pkt[UDP].dport)
                    l4_proto = "UDP"
                elif SCTP in pkt:
                    src_port = int(pkt[SCTP].sport)
                    dst_port = int(pkt[SCTP].dport)
                    l4_proto = "SCTP"
                else:
                    src_port = -1
                    dst_port = -1
                    l4_proto = "unknown"  # arp, lldp
            except Exception:   #(protocols that don't reach transport layer do not use port numbers)
                src_port = -1
                dst_port = -1
                l4_proto="unknown" #arp, ...

            #get application layer protocol from port number
            app_proto = get_app_proto(src_port,dst_port)

            # frame length
            frame_len = len(pkt.original)

            #Ethernet / IP / TCP / Raw
            #-> pkt.payload = IP Layer, Pkt[IP].payload = TCP layer etc

            #for packets without application layer
            if l4_proto=="unknown":
                header_len=frame_len    #whole packet is header
                app_len=0
            else: #packets with application layer
                header_len,app_len=header_and_payload_len(pkt)


            #host pairs (sorted concatination of src and dst ip address)
            if not src_ip == "":
                pair_id = host_pair_id(src_ip, dst_ip)
            else:
                pair_id = host_pair_id(src_mac, dst_mac)

            #host pair and application layer protocol key
            proto_pair_key = f"{pair_id}__{app_proto}"

            #calculate iat for each host pair
            if pair_id:
                if pair_id in last_ts_by_pair:
                    iat_pair = ts - last_ts_by_pair[pair_id]
                else:
                    iat_pair = pd.NA
                last_ts_by_pair[pair_id] = ts #stores last timestamp for that pair ID in dictionary


            #calculate iat for pairs with same app protocol
            if pair_id and app_proto:
                if proto_pair_key in last_ts_by_proto_pair:
                    iat_proto_pair = ts - last_ts_by_proto_pair[proto_pair_key]
                else:
                    iat_proto_pair=pd.NA
                last_ts_by_proto_pair[proto_pair_key] = ts

            records.append({
                "timestamp": ts_rel,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "src_port": src_port,
                "dst_port": dst_port,
                "l4_proto": l4_proto,
                "app_proto": app_proto,
                "frame_len": frame_len,
                "total_header_len": header_len,
                "app_payload_len": app_len,
                "pair_id": pair_id,
                "iat_pair": iat_pair, # is this the inter-arrival time between packets? 
                "iat_proto_pair": iat_proto_pair, # what is the difference between this and the previous one?
                "label_attack": int(attack),
                "filename": filename
            })


            #print(ts, frame_len, src_ip, dst_ip, src_port, dst_port, l4_proto, app_proto, header_len, app_len, iat_pair, iat_proto_pair)
    return pd.DataFrame(records)






