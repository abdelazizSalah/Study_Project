import gc

import pandas as pd

#todo: create one csv file for attacks and one for control, so that existing functions can be used
def electra_df_extract_values(df):
    records=[]

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
        "iat_pair": iat_pair,  # is this the inter-arrival time between packets?
        "iat_proto_pair": iat_proto_pair,  # what is the difference between this and the previous one?
        "label_attack": int(attack),
        "filename": filename
    })


    #write directly to csv
    return

csv_path="/home/dW5kZWFk/Downloads/electra_s7comm.csv"

def read_electra():
    usecols = ["Time", "smac", "dmac", "sip", "dip", "request", "fc", "error", "address", "data", "label"]
    dtypes = {
        "smac": "string", "dmac": "string",
        "sip": "string", "dip": "string",
        "request": "Int8", "fc": "Int16", "error": "Int8", "address": "Int32",
        "data": "string", "label": "string",
    }

    chunksize=500_000

    chunks = pd.read_csv(csv_path, chunksize=chunksize,
                             usecols=usecols, sep=",", dtype=dtypes,
                             low_memory=False, on_bad_lines="skip")

    first_chunk=next(chunks)
    print(first_chunk)


read_electra()