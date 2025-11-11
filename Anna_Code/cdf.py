import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_statistics import preprocess

#
#takes list of values [1,3,5,7]
def compute_manual_cdf(data):

    # Remove NaN values and convert to list
    clean_data = []
    for x in data:
        if pd.notna(x):
            clean_data.append(x)


    n = len(clean_data)

    if n == 0:
        return [], []

    # Sort the data in ascending order
    sorted_vals = sorted(clean_data)

    #[1,2,3,4,5]
    #[1/5,2/5,3/5,4/5,5,5]
    # Compute cdf
    y = []
    for i in range(1, n + 1):
        fraction = i / n #divide index by total number of elements
        y.append(fraction)

    return sorted_vals, y



#can be used with header length or payload length
#plot for each application layer protocol
#sub_df is control or attack df
def plot_cdf_per_proto(sub_df, column, dataset_label):
    """
    Plots manual CDFs for one numeric column (header or payload length)
    with one line per application-layer protocol.
    """

    if column=="app_payload_len":
        column_label="Application Payload Length"
    else:
        column_label="Total Header Length"

    #control ds
    protocols = sub_df['app_proto'].unique()

    plt.figure(figsize=(8, 6))

    for proto in protocols:
        subset = sub_df[sub_df['app_proto'] == proto] #subset of df with one protocol only
        if subset.empty:
            continue
        x, y = compute_manual_cdf(subset[column]) #column: header_length or payload_length
        plt.plot(x, y, label=proto)

    plt.xlabel(f"{column_label} (bytes)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of {column_label} per Application Protocol for {dataset_label} ")
    plt.legend(title="App Protocol", loc="lower right")
    plt.grid(True)
    plt.tight_layout()


    plt.savefig(f"cdf_{column}_{dataset_label}.png")

    return


#for each value x in the dataset it shows how likely it is that a value <=x occurs
#eg for x=5: add all datapoints that have smaller or equal values together and divide by total number of values
def create_cdf_plots_task1d(df, output_dir, dataset_name):
    """Plot CDFs for header length and application payload length per application-layer protocol,
       and save them into the given output directory.
    """

    # filter out packets without application layer protocol
    df = df[df['app_proto'] != 'unknown']

    # header length
    plot_cdf_per_proto(df[df['label_attack'] == 0], "total_header_len", f"Control Dataset in {dataset_name}")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_control_header_len_cdf.png"), bbox_inches="tight")
    plt.close()

    plot_cdf_per_proto(df[df['label_attack'] == 1], "total_header_len", f"Attack Dataset in {dataset_name}")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_attack_header_len_cdf.png"), bbox_inches="tight")
    plt.close()

    # application payload length
    plot_cdf_per_proto(df[df['label_attack'] == 0], "app_payload_len", f"Control Dataset in {dataset_name}")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_control_payload_len_cdf.png"), bbox_inches="tight")
    plt.close()

    plot_cdf_per_proto(df[df['label_attack'] == 1], "app_payload_len", f"Attack Dataset in {dataset_name}")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_attack_payload_len_cdf.png"), bbox_inches="tight")
    plt.close()

    return 

