import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Anna_Code.get_statistics import preprocess


#takes list of values [1,3,5,7]
def compute_manual_cdf(data):

    # Remove NaN values and convert to list
    clean_data = [x for x in data if pd.notna(x)]
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

    #control ds
    protocols = sub_df['app_proto'].unique()

    plt.figure(figsize=(8, 6))

    for proto in protocols:
        subset = sub_df[sub_df['app_proto'] == proto] #subset of df with one protocol only
        if subset.empty:
            continue
        x, y = compute_manual_cdf(subset[column]) #column: header_length or payload_length
        if proto=="s7comm":
            print(f"proto:{proto}-{x}")
        plt.plot(x, y, label=proto)

    plt.xlabel(f"{column} (bytes)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of {column} per Application Protocol for {dataset_label} ")
    plt.legend(title="App Protocol", loc="lower right")
    plt.grid(True)
    plt.tight_layout()


    plt.savefig(f"cdf_{column}_{dataset_label}.png")

    #attack ds


#for each value x in the dataset it shows how likely it is that a value <=x occurs
#eg for x=5: add all datapoints that have smaller or equal values together and divide by total number of values
def create_cdf_plots_task1d(df):
    """Plot CDS for header length  per application layer protocol for control dataset and attack dataset.
        Plot CDF for application payload length per application layer protocol for control dataset and attack dataset. """

    #filter out packets without application layer protocol
    df = df[df['app_proto'] != 'undetected:-1:-1']

    #header length
    plot_cdf_per_proto(df[df['label_attack'] == 0],"total_header_len","Control Dataset in QUT_S7Comm")
    plot_cdf_per_proto(df[df['label_attack'] == 1],"total_header_len","Attack Dataset in QUT_S7Comm")

    #application payload length
    plot_cdf_per_proto(df[df['label_attack'] == 0], "app_payload_len", "Control Dataset in QUT_S7Comm")
    plot_cdf_per_proto(df[df['label_attack'] == 1], "app_payload_len", "Attack Dataset in QUT_S7Comm")
    return 0

