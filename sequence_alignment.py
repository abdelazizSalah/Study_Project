import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing.print_coercion_tables import print_new_cast_table
from Assignment2.Task1.kmeans import kmeans_iat_clusters, plot_iat_kmeans_clusters
from Bio.Align import PairwiseAligner

def calculate_profile_score(profile_column_residues, residue_s3, match_score, mismatch_score, gap_penalty):
    """
    Calculates the Sum-of-Pairs (SP) score for aligning one residue (from S3)
    against one column of the profile (S1 and S2 residues/gaps).

    NOTE: This is a simplified SP score calculation.
    """
    N = len(profile_column_residues)  # Should be 2 (S1 and S2)
    total_score = 0

    # Iterate through all sequences currently in the profile (S1 and S2)
    for res_profile in profile_column_residues:

        # ⚠️ CRITICAL: Handle the gap character in the profile
        # (assuming 0 represents a gap/insertion in the byte list)
        if res_profile == 0:
            # If S1 or S2 had a gap, score the S3 residue vs a gap penalty
            # Since S3 is the one being matched, we use the gap penalty related to S3 matching a gap
            total_score += gap_penalty
        else:
            # Score S3 residue vs the S1/S2 residue
            if res_profile == residue_s3:
                total_score += match_score
            else:
                total_score += mismatch_score

    # Return the average score for normalization (Sum-of-Pairs / N)
    return total_score / N


def align_sequence_to_profile(alignment_s1_s2, seq3_bytes, match=5, mismatch=-1, gap_open=-3, gap_extend=-0.5):
    """
    Performs profile-sequence alignment using a custom scoring matrix based on SP scores.
    """

    # 1. Extract S1 and S2 from the previous alignment object
    # The alignment object gives you the strings/lists including gaps.
    S1_aligned = list(alignment_s1_s2.seqA)  # S1 with gaps
    S2_aligned = list(alignment_s1_s2.seqB)  # S2 with gaps

    profile_columns = []
    # Combine S1_aligned and S2_aligned into columns (the Profile)
    # We iterate over the length of the aligned sequences
    for i in range(len(S1_aligned)):
        # Assuming the Biopython alignment gives you the original characters (or bytes)
        # You'll need to handle the gap character used by Biopython (often '-')
        # For simplicity, let's assume S1_aligned/S2_aligned are lists of bytes/ints,
        # and a placeholder (like 0) marks a gap.
        profile_columns.append([S1_aligned[i], S2_aligned[i]])

    # 2. Build the Custom Scoring Matrix for Profile vs S3
    # M = len(profile_columns), L = len(seq3_bytes)
    # This is where the standard dynamic programming implementation would be replaced.
    # Since we can't easily modify Biopython's internal DP matrix,
    # the practical implementation often uses a dedicated MSA library (like Clustal)
    # or you implement the full DP algorithm yourself.

    # 💡 A common workaround for illustration (though not fully correct dynamic programming):
    # This simplified version just uses the standard aligner but with a trickier scoring.
    # Due to the complexity of integrating custom column scores into a standard DP algorithm
    # like Biopython's, it's often better to use a library designed for MSA (like MUSCLE/Clustal).

    # ⚠️ Warning: Since we cannot easily implement the custom scoring function
    # into PairwiseAligner, this function is **conceptual**.
    # For a *true* implementation, you would need to write the full DP loop (initialization,
    # recurrence, and traceback) and use `calculate_profile_score` in the recurrence step.

    # If you were to use a full DP implementation, the final alignment would be:
    # final_alignment_s1_s2_s3 = traceback_results

    print("\n⚠️ Profile-Sequence Alignment is Conceptual Here.")
    print("A full implementation requires replacing the Biopython PairwiseAligner with a custom")
    print("Dynamic Programming loop that uses the `calculate_profile_score` for matches.")

    return None  # Return the new MSA (S1, S2, S3)

############################################################################################################
#input: "02 F0 A3" ...
#output [2,240,...]
def hex_to_bytes_list(hex_string):
    hex_string = hex_string.replace(" ", "")
    bytes_list=[ ]
    for position in range(0, len(hex_string), 2):
        two_characters = hex_string[position: position + 2]

        # Step 4: Convert those 2 characters from hex to an integer
        byte_value = int(two_characters, 16)

        # Step 5: Add the byte (integer) to our list
        bytes_list.append(byte_value)
        #bytes_object=bytes(bytes_list)
        #s1 = bytes_object.decode('latin-1')
    return bytes_list


def nw_score_modern(seq1, seq2, match=5, mismatch=-1, gap_open=-3, gap_extend=-0.5):
    """
    Computes Needleman–Wunsch alignment score using the modern Biopython PairwiseAligner.
    seq1, seq2 can be lists of integers (byte values).
    """

    # 1. Initialize the Aligner object
    aligner = PairwiseAligner()

    # 2. Set the scoring parameters (Needleman-Wunsch is global)
    aligner.mode = 'global'
    aligner.match_score = match
    aligner.mismatch_score = mismatch

    # Use Affine Gap Penalties (Start cost and Extend cost)
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    # 3. Calculate the score (passing the lists of integers directly is supported!)
    # seq1 and seq2 are passed here as lists of integers: [2, 240, 163, ...]
    score = aligner.score(seq1, seq2)

    #compute theoretical best and worst possible score
    L_min = max(len(seq1), len(seq2))
    # Handle empty sequences
    if L_min == 0:
        return 1.0  # Max dissimilarity if at least one sequence is empty

    S_max = L_min * match

    # 3. Calculate Normalized Dissimilarity (D_norm)
    # The score is capped at S_max, so D_norm will be between 0 and 1 (or slightly negative)
    D_norm = 1.0 - (score / S_max)

    # Optional: Ensure the result is strictly between 0 and 1
    # D_norm might be slightly less than 0 if the actual S_raw is slightly higher
    # than S_max due to overlapping gap penalties.
    D_norm = min(max(D_norm, 0.0), 1.0)

    return D_norm



def get_distance_score_matrix(list_of_sequences):

    n=len(list_of_sequences)

    matrix = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):   #start from i to have no duplicate calculations
            if i==j:
                matrix[i][j] = math.nan #insert NA
            else:
                #convert to byte list
                seq1 = hex_to_bytes_list(list_of_sequences[i])
                seq2 = hex_to_bytes_list(list_of_sequences[j])

                # Compute distance once
                dist = nw_score_modern(seq1, seq2)

                # Store in both symmetric positions
                matrix[i][j] = dist
                matrix[j][i] = dist

    return matrix


#optional
def verify_distance_matrix(distance_score_matrix, sequence_list_as_hex):
    n=len(distance_score_matrix[1])
    for i in range(n):
        for j in range(n):
            if(distance_score_matrix[i][j]==1.0):
                print(distance_score_matrix[i][j])
                print(sequence_list_as_hex[i])
                print(sequence_list_as_hex[j])

    return


def get_full_alignment(seq1, seq2, match=5, mismatch=-1, gap_open=-3, gap_extend=-0.5):
    """
    Performs a global alignment (Needleman-Wunsch) and returns the alignment objects.
    seq1 and seq2 are lists of integers (byte values).
    """

    # 1. Initialize and configure the aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    # Use the proven, sensitive scoring parameters
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    # 2. Get all optimal alignments
    alignments = aligner.align(seq1, seq2)

    # 3. Return the results
    return alignments


def start_sequence_alignment(df):
    sequence_list_as_hex=df["data"].tolist()

    distance_score_matrix=get_distance_score_matrix(sequence_list_as_hex)
    print(distance_score_matrix)
    seq1_bytes=hex_to_bytes_list(sequence_list_as_hex[2])
    seq2_bytes=hex_to_bytes_list(sequence_list_as_hex[3])
    seq3_bytes = hex_to_bytes_list(sequence_list_as_hex[4])
    #print some examples for the score
    #for i in range(n):
    #    print(sequence_list_as_hex[i])
    #    print(sequence_list_as_hex[i+1])
    #    print(matrix[i][i+1])
    #

    #test the actual aligner idk hahaha bye
    alignments = get_full_alignment(seq1_bytes, seq2_bytes)
    best_alignment_s1_s2=next(alignments)

    # Print the resulting alignments
    try:
        print("\n✅ Only the first optimal alignment:")
        print(best_alignment_s1_s2)
    except StopIteration:
        print("No alignments found.")

    new_msa_s1_s2_s3 = align_sequence_to_profile(best_alignment_s1_s2, seq3_bytes)
    print(new_msa_s1_s2_s3)
    return 0


##############################################################################################################
def group_into_communication_sessions(df, threshold):
    #if iat is bigger than threshold or session_id is different to previous one -> start new group

    group_ids = []
    prev_session_id = None
    current_group = 0

    # iterate rows in order
    for _, pkt in df.iterrows():

        # condition: start a new group
        if (pkt["session_id"] != prev_session_id) or (pkt["iat_session_pair"] > threshold) or (pkt["iat_session_pair"].isna()): #todo: does this make sense?
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

    # cumulative sum of True/False → 1,2,3,... group ids
    df["group_id"] = new_group.cumsum()

    return df


#largest gap method
#sort them from smallest to largest
#find the biggest jump in difference
def iat_gap_threshold(df):
    iats = df["iat_session_pair"].dropna().to_numpy()
    # only consider the ones with smallest threshold
    filtered_iats = iats[iats < 0.3]    #todo: proof that meaningfull region is below this value
    iats_sorted = np.sort(filtered_iats)

    diffs = np.diff(iats_sorted)

    # position of the largest jump
    idx = np.argmax(diffs)

    threshold = iats_sorted[idx]
    small_cluster = iats_sorted[:idx+1]
    large_cluster = iats_sorted[idx+1:]

    return threshold, small_cluster, large_cluster


def find_threshold_iat_kmeans(df):
    #iat_values=df["iat_session_pair"].dropna()

    iats = df["iat_session_pair"].dropna().to_numpy()
    filtered_iats = iats[iats < 0.3]

    #cluster all iats
    iats, labels, centers, threshold = kmeans_iat_clusters(filtered_iats, 2)
    plot_iat_kmeans_clusters(iats, labels, centers, threshold, "kmeans_clusters_filtered.png")




