import os
from file_helper_t2 import list_files_by_filetype, save_df_to_csv
from scapy.all import PcapReader, TCP, IP
import pandas as pd


#checks if port number corresponds to the one for s7comm
def check_if_app_proto_is_s7comm(srcport,dstport):
    ports=[srcport,dstport]

    for p in ports:
        if p==102:
            return True

    return False



#creates session_id that identifies communication session
def create_session_id(src_ip, dst_ip, src_port, dst_port):
    a = (src_ip, str(src_port))
    b = (dst_ip, str(dst_port))
    pair = sorted([a, b])
    return f"{pair[0][0]}:{pair[0][1]}__{pair[1][0]}:{pair[1][1]}"
#IP1:102__IP2:B = IP2:B__IP1:102 (exchange src and dst)
#what doesnt work: IP1:102__IP2:B = IP1:C__IP2:102


#input: pcap file
#output: dataframe with extracted value per packet
def pcap_extract_values(pcap_path):

    records = []
    last_ts_by_session_pair = {}

    first_ts=None

    with (PcapReader(pcap_path) as pcap):

        for pkt in pcap:

            #check if s7comm, if not -> skip:
            try:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
            except Exception as e:  #missing IP layer triggers exception
                continue  #s7commm would have IP layer

            if TCP in pkt: #s7comm only runs on tcp
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
            else:
                continue

            if not  check_if_app_proto_is_s7comm(src_port,dst_port):
                continue


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


            #session ips
            session_id = create_session_id(src_ip, dst_ip, src_port, dst_port)


            #calculate iat for each host pair
            if session_id in last_ts_by_session_pair:
                iat_session_pair = ts - last_ts_by_session_pair[session_id]
            else:
                iat_session_pair = pd.NA
            last_ts_by_session_pair[session_id] = ts #stores last timestamp for that pair ID in dictionary
            raw_bytes = bytes(pkt[TCP].payload)

            #skip empty payloads
            if not raw_bytes:
                continue
            payload_hex = raw_bytes.hex(" ").upper()  # "03 00 00 16 ..." (space-separated upper case hex string)

            records.append({
                "timestamp": ts_rel,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "session_id": session_id,
                "iat_session_pair": iat_session_pair,
                "data": payload_hex
            })


            #print(ts, frame_len, src_ip, dst_ip, src_port, dst_port, l4_proto, app_proto, header_len, app_len, iat_pair, iat_proto_pair)
    return pd.DataFrame(records)



#used for QUT_S7Comm
def preprocess_dataset(input_path_pcap, output_csv_file):

    pcap_files=list_files_by_filetype(input_path_pcap,"pcap")

    first_control_file=1
    for path in pcap_files:

        filename = os.path.basename(path)
        try:
            write_header = first_control_file

            if first_control_file:
                file_mode = 'w'
                first_control_file=0
            else: file_mode='a'

            df_pcap=pcap_extract_values(path)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=write_header)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nCreated CSV from PCAP files\n\n")

    return 0
