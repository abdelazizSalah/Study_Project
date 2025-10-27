import os
from traceback import print_stack

import numpy as np
from scapy.all import PcapReader, rdpcap, IP, IPv6, TCP, UDP
import pandas as pd
from scapy.layers.sctp import SCTP
from known_ports import KNOWN_PORTS


"""
Extract all the important values from a pcap file and summarize them in a dataframe.
"""


#application layer protocol
def get_app_proto(srcport,dstport):
    #we assume that only either the dstport or the srcport is well known
    ports=[srcport,dstport]

    for p in ports:
        if p in KNOWN_PORTS:
            return KNOWN_PORTS[p]

    #todo complete mapping
    #toDo what if there is no application layer -> return "none"
    return(f"undetected:{srcport}:{dstport}")


#length of payload and header of each packet
def header_and_payload_len(pkt):
    #toDo: fix (currently everything inside l4 is counted as "payload")

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
def host_pair_id(src_ip,dst_ip):
    # replace with empty string if None
    a = src_ip or ""
    b = dst_ip or ""
    return "__".join(sorted([a, b]))



#filename
#pkt index = timestamp
#src ip
#dst ip
#src port
#dst port
#packet length (Frame length)

#header length
#appliction payload length
#application layer protocol
#transport layer protocol
#host pairs
#inter-arrival time -> consecuticve packets (timestamps?)
#payload content?
#boolean attack

#returns dataframe object for a pcap
def pcap_extract_values(pcap_path, attack):

    records = []
    last_ts_by_pair = {}
    last_ts_by_proto_pair = {}
    pkt_index=0
    filename=os.path.basename(pcap_path)
    with (PcapReader(pcap_path) as pcap):
        for pkt in pcap:
            # timestamp (always try to keep)
            try:
                ts = float(pkt.time)
            except Exception:
                # if timestamp missing, set 0.0 (row still recorded)
                ts = 0.0

            try:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
            except Exception:
                src_ip = ""
                dst_ip = ""


            #(protocols that don't reach transport layer do not use port numbers)
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
            except Exception:
                src_port = -1
                dst_port = -1
                l4_proto="unknown" #arp, ...

            app_proto = get_app_proto(src_port,dst_port)


            # frame length
            frame_len = len(pkt.original)

            #Ethernet / IP / TCP / Raw
            #-> pkt.payload = IP Layer, Pkt[IP].payload = TCP layer etc

            #for packets without application layer -> header len is 0
            if l4_proto=="unknown":
                header_len=frame_len
                app_len=0
            else:
                header_len,app_len=header_and_payload_len(pkt)

            if header_len is None or header_len < 0 or header_len > frame_len:
                raise ValueError(f"Invalid total_header_len: {header_len} ")

            #host pairs and inter arrival time
            pair_id = host_pair_id(src_ip, dst_ip)

            #for calculations per proto-hostpair combinarion
            proto_pair_key = f"{pair_id}__{app_proto}"

            #calculate iat for pair only
            if pair_id:
                if pair_id in last_ts_by_pair:
                    iat_pair = ts - last_ts_by_pair[pair_id]
                else:
                    iat_pair = pd.NA
                last_ts_by_pair[pair_id] = ts #stores last timestamp for that pair ID (pcap files are sorted by timestamp)


            #calculate iat for pairs with same app protocol
            if pair_id and app_proto:
                if proto_pair_key in last_ts_by_proto_pair:
                    iat_proto_pair = ts - last_ts_by_proto_pair[proto_pair_key]
                else:
                    iat_proto_pair=pd.NA
                last_ts_by_proto_pair[proto_pair_key] = ts

            records.append({
                "timestamp": ts,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "l4_proto": l4_proto,
                "app_proto": app_proto,
                "frame_len": frame_len,
                "total_header_len": header_len,
                "app_payload_len": app_len,
                "pair_id": pair_id,
                "iat_pair": iat_pair,
                "iat_proto_pair": iat_proto_pair,
                "label_attack": int(attack),
                "filename": filename
            })


            #print(ts, frame_len, src_ip, dst_ip, src_port, dst_port, l4_proto, app_proto, header_len, app_len, iat_pair, iat_proto_pair)
    return pd.DataFrame(records)





#only for verification
#def read_pcap_as_scapy_object(pcap_path):
#    packets = rdpcap(pcap_path)
#    print(f"Total packets: {len(packets)}")
#    return packets

