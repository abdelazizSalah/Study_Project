import gc

import pandas as pd

from Anna_Code.file_helper import save_df_to_csv
from process_pcap import host_pair_id

#todo: create one csv file for attacks and one for control, so that existing functions can be used
#
def electra_df_extract_values(df):

    records = []
    last_ts_by_pair = {}

    header_len=14+20+20+4+3 #assume ethernet+ip+tcp+rfc+cotp
    for index, row in df.iterrows():
        src_ip=row["sip"]
        dst_ip=row["dip"]

        src_mac=row["smac"]
        dst_mac=row["dmac"]
        ts=row["Time"]

        # host pairs and inter arrival time
        if not src_ip == "":
            pair_id = host_pair_id(src_ip, dst_ip)
        else:
            pair_id = host_pair_id(src_mac, dst_mac)

        proto_pair_key = f"{pair_id}__s7comm"

        #calculate iat for pair only:
        if pair_id:
            if pair_id in last_ts_by_pair:
                iat_pair = ts - last_ts_by_pair[pair_id]
            else:
                iat_pair = pd.NA
            last_ts_by_pair[pair_id] = ts

        #in electra the app protocol is always s7comm, so the following is true;
        iat_proto_pair=iat_pair

        attack=1
        if row["label"]=="NORMAL":
            attack=0

        app_len=int(row["data"])+10 #payload length + s7comm header

        records.append({
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "l4_proto": "TCP",
            "app_proto": "s7comm",
            "frame_len": header_len+app_len,
            "total_header_len": header_len, #standardized for expected stack:  Ethernet → IP → TCP → S7Comm (excluding S7Comm)
            "app_payload_len": app_len, #s7comm header + payload to be comparable with QUT statistics
            "pair_id": pair_id,
            "iat_pair": iat_pair,  # is this the inter-arrival time between packets?
            "iat_proto_pair": iat_proto_pair,  # what is the difference between this and the previous one?
            "label_attack": attack,
            "filename": "electra_s7comm.csv"
        })

    #toDo: removed src_port and dst_port: fix in getStatistics!
    #write directly to csv
    return pd.DataFrame(records)


def read_electra_to_csv(csv_input_path,csv_output_path):
    usecols = ["Time", "smac", "dmac", "sip", "dip", "request", "fc", "error", "address", "data", "label"]
    dtypes = {
        "smac": "string", "dmac": "string",
        "sip": "string", "dip": "string",
        "request": "Int8", "fc": "Int16", "error": "Int8", "address": "Int32",
        "data": "string", "label": "string",
    }

    chunksize=500_000

    chunks = pd.read_csv(csv_input_path, chunksize=chunksize,
                             usecols=usecols, sep=",", dtype=dtypes,
                             low_memory=False, on_bad_lines="skip")

    first_chunk=next(chunks)
    df=electra_df_extract_values(first_chunk)
    save_df_to_csv(df, csv_output_path, mode='w',header=1)
    return 0
    print("")
