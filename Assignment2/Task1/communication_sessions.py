from Assignment2.Task1.sequence_alignment import filter_s7_packets
import numpy as np


##############################################################################################################
def group_into_communication_sessions(df, threshold):
    #if iat is bigger than threshold or session_id is different to previous one -> start new group

    group_ids = []
    prev_session_id = None
    current_group = 0

    # iterate rows in order
    for _, pkt in df.iterrows():

        # different session ID / iat bigger than threshold / start of new file -> new communication session group!
        if (pkt["session_id"] != prev_session_id) or (pkt["iat_session_pair"] > threshold) or (pkt["iat_session_pair"].isna()):
            current_group += 1
            print("new group")

        group_ids.append(current_group)
        prev_session_id = pkt["session_id"]

    df["group_id"] = group_ids
    return df


def group_into_communication_sessions_optimized(df, threshold):
    # make sure packets are in the right order
    # (adjust "Time" to your actual time column name)
    df = df.sort_values(["session_id", "timestamp"]).copy()

    # True where a new group should start
    new_group = (
        (df["session_id"] != df["session_id"].shift()) |
        (df["iat_session_pair"] > threshold) |
        (df["iat_session_pair"].isna())
    )

    df["group_id"] = new_group.cumsum()

    return df


#largest gap method
#sort them from smallest to largest
#find the biggest jump in difference
def iat_gap_threshold(df):
    iats = df["iat_session_pair"].dropna().to_numpy()
    # only consider the ones with smallest threshold
    filtered_iats = iats[iats < 1.0]    #todo: proof that meaningfull region is below this value
    iats_sorted = np.sort(filtered_iats)

    diffs = np.diff(iats_sorted)

    # position of the largest jump
    idx = np.argmax(diffs)

    threshold = iats_sorted[idx]
    small_cluster = iats_sorted[:idx+1]
    large_cluster = iats_sorted[idx+1:]

    return threshold, small_cluster, large_cluster


def create_communication_sessions(df):
    df_s7 = filter_s7_packets(df)   #ensure df only contains s7comm packets

    #find threshold
    threshold, small_cluster, large_cluster =iat_gap_threshold(df_s7)

    df=group_into_communication_sessions_optimized(df_s7, threshold)
    return df