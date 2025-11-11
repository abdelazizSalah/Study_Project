import gc

import numpy as np
import pandas as pd

from file_helper import read_df_from_csv, save_df_as_parquet



HEADER_LEN = 14 + 20 + 20  # Ethernet+IP+TCP

#using vectorization instead of iterrows for speed improvement
def electra_df_extract_values_optimized_vectorization(df):

    # every entry has IP address (no MAC address used)
    a = df["sip"].fillna("") #replace missing ip
    b = df["dip"].fillna("")

    #host pair ID
    first = np.where(a <= b, a, b)  #if a<=b, pick a as first values
    second = np.where(a <= b, b, a) #if a <=b pick b as second value
    df["pair_id"] = pd.Series(first, index=df.index) + "__" + pd.Series(second, index=df.index)

    # iat
    df["iat_pair"] = df.groupby("pair_id")["Time"].diff()
    #diff[i] = current_row_value - previous_row_value

    #if rows are identical they stay unless deduplicated

    # Features
    app_len = df["data"].astype("int32") + 10

    df_out = pd.DataFrame({
        "timestamp": df["Time"],
        #"src_ip": df["sip"],
        #"dst_ip": df["dip"],
        #"src_mac": df["smac"],
        #"dst_mac": df["dmac"],
        "l4_proto": "TCP",
        "app_proto": "s7comm",
        "frame_len": HEADER_LEN + app_len,
        "total_header_len": HEADER_LEN,
        "app_payload_len": app_len,
        "pair_id": df["pair_id"],
        "iat_pair": df["iat_pair"],
        "iat_proto_pair": df["iat_pair"],  # identisch bei dir
        "label_attack": (df["label"] != "NORMAL"),
        "filename": "electra_s7comm.csv"
    })
    return df_out



def read_electra_create_parquet(csv_input_path,csv_output_path):
    df_in=read_df_from_csv(csv_input_path)
    df_out = electra_df_extract_values_optimized_vectorization(df_in)
    save_df_as_parquet(df_out, csv_output_path)
    return 0

