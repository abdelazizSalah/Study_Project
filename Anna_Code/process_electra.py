import gc

import pandas as pd

#todo: create one csv file for attacks and one for control, so that existing functions can be used
#
def electra_df_extract_values(df):
    records=[]

    filename="electra_s7comm.csv"
    for index, row in df.iterrows():
        print(index, row['column_name'])
        sip=row["sip"]
        dip=row["dip"]

        #if no IP address use mac address instead
        if pd.isna() or str().strip() == '':
            sip=row["smac"]
            dip=row["dmac"]

    app_len=10+12
    records.append({
        "timestamp": df["Time"],
        "src_ip": row["sip"],
        "dst_ip": row["sip"],
        "src_mac": row["smac"],
        "dst_mac": row["dmac"],
        "src_port": src_port,   #none
        "dst_port": dst_port,   #none
        "l4_proto": "TCP",
        "app_proto": "s7comm",
        "frame_len": 68+app_len,
        "total_header_len": 68, #standardized for expected stack:  Ethernet → IP → TCP → S7Comm (excluding S7Comm)
        "app_payload_len": app_len, #s7comm header + payload to be comparable with QUT statistics
        "pair_id": pair_id,
        "iat_pair": iat_pair,  # is this the inter-arrival time between packets?
        "iat_proto_pair": iat_proto_pair,  # what is the difference between this and the previous one?
        "label_attack": int(attack),
        "filename": filename
    })


    #write directly to csv
    return


def read_electra(csv_path):
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

    for i in first_chunk:
        print(first_chunk["Time"])
        if (pd.isna(first_chunk["sip"] ) or str(first_chunk["sip"] ).strip() == ''):
            print(first_chunk["dmac"])
