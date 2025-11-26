import math
from Bio.Align import PairwiseAligner
from Assignment2.Task1.sequence_alignment_print import show_alignment_block


#filter out everything that doesnt start with 0300 (not s7comm)
#filter out tiny packets < (not s7comm)
def filter_s7_packets(df):
    data_col="data"
    indices_to_keep = []
    cleaned_hex_by_idx = {}

    for idx, seq in df[data_col].items():

        if not isinstance(seq, str):
            continue

        hex_clean = seq.replace(" ", "").lower()

        #tpkt header should be there
        if not hex_clean.startswith("0300"):
            continue

        #convert hex string into byte list
        try:
            byte_list = []
            hex_string = hex_clean

            for i in range(0, len(hex_string), 2):
                # extract current byte fro string
                hex_segment = hex_string[i:i + 2]

                # hex to byte
                byte_value = int(hex_segment, 16)

                byte_list.append(byte_value)
        except ValueError:
            continue


        #remove TPKT + COTP part (first 7 bytes) 03 00 ...
        s7_bytes = byte_list[7:]

        # s7comm packet should at least be 12 bytes long
        if len(s7_bytes) < 12:
            continue

        # skip packets that do not startwith 0x32 (not s7comm)
        if s7_bytes[0] != 0x32:
            continue

        indices_to_keep.append(idx)

        s7_hex_spaced = " ".join(f"{b:02x}" for b in s7_bytes)
        cleaned_hex_by_idx[idx] = s7_hex_spaced

    # Nur die gültigen Zeilen behalten
    df_out = df.loc[indices_to_keep].copy()

    # data_col durch S7-PDU (ab 0x32) ersetzen
    df_out[data_col] = df_out.index.map(cleaned_hex_by_idx.get)

    return df_out


#input: "02 F0 A3" ...
#output [2,240,...]
def hex_to_bytes_list(hex_string):
    hex_string = hex_string.replace(" ", "")
    bytes_list=[ ]
    for position in range(0, len(hex_string), 2):
        two_characters = hex_string[position: position + 2]

        #Convert those 2 characters from hex to an integer
        byte_value = int(two_characters, 16)

        #  Add the byte (integer) to our list
        bytes_list.append(byte_value)
        #bytes_object=bytes(bytes_list)
        #s1 = bytes_object.decode('latin-1')
    return bytes_list


#use NW wunsch score for distance score matrix
def nw_score_modern(seq1, seq2, match=5, mismatch=-1, gap_open=-3, gap_extend=-0.5):

    # Initialize the Aligner object
    aligner = PairwiseAligner()

    # Set the scoring parameters (Needleman-Wunsch is global)
    aligner.mode = 'global'
    aligner.match_score = match
    aligner.mismatch_score = mismatch

    # Use Affine Gap Penalties (Start cost and Extend cost)
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    #calculate the score (passing the lists of integers directly is supported!)
    # seq1 and seq2 are passed here as lists of integers: [2, 240, 163, ...]
    score = aligner.score(seq1, seq2)

    #compute theoretical best and worst possible score
    L_min = max(len(seq1), len(seq2))
    # Handle empty sequences
    if L_min == 0:
        return 1.0  # Max dissimilarity if at least one sequence is empty

    S_max = L_min * match

    # Calculate Normalized Dissimilarity (D_norm)
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


def biopython_alignment_to_lists(alignment):


    rowA = alignment[0]
    rowB = alignment[1]

    alignedA = []
    alignedB = []

    for a in rowA:
        if a is None or a == "-" or a == b"-":
            alignedA.append(None)
        else:
            alignedA.append(int(a))

    for b in rowB:
        if b is None or b == "-" or b == b"-":
            alignedB.append(None)
        else:
            alignedB.append(int(b))

    return alignedA, alignedB


def get_full_alignment(seq1, seq2, match=5, mismatch=-1, gap_open=-3, gap_extend=-0.5):


    aligner = PairwiseAligner()
    aligner.mode = 'global'

    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    alignments = aligner.align(seq1, seq2)

    return alignments


def find_most_similar_pair(dist_matrix):
    n = len(dist_matrix)
    best = None
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            current_dist=dist_matrix[i][j]
            if dist_matrix[i][j] < min_dist and current_dist>0.0:   #ignore sequences that are similair
                min_dist = dist_matrix[i][j]
                best = (i, j)
    return best, min_dist


def expand_alignment_from_ref(old_alignment, old_ref, new_ref):

    expanded = [[] for _ in old_alignment]

    i_old = 0
    len_old = len(old_ref)

    for val in new_ref:
        if val is None:

            for row in expanded:
                row.append(None)
        else:

            if i_old < len_old:
                for row_idx, row in enumerate(old_alignment):

                    if i_old < len(row):
                        row_val = row[i_old]
                    else:
                        row_val = None
                    expanded[row_idx].append(row_val)
                i_old += 1
            else:
                for row in expanded:
                    row.append(None)

    if i_old != len_old:
        print(f"[WARN] expand_alignment_from_ref: i_old={i_old}, len(old_ref)={len_old}")

    return expanded


#choose next sequence from distance matrix
#highest score towards one of the already used sequences
def find_next_sequence(dist_matrix, used_indices):
    n = len(dist_matrix)
    remaining = [i for i in range(n) if i not in used_indices]
    if not remaining:
        return None

    best = None
    best_dist = float('inf')
    for r in remaining:
        d = min(dist_matrix[r][u] for u in used_indices)
        if d < best_dist:
            best_dist = d
            best = r
    return best


def build_progressive_alignment(sequence_list_as_hex, distance_score_matrix):
    n = len(sequence_list_as_hex)

    # find most similair pair
    (i, j), min_dist = find_most_similar_pair(distance_score_matrix)

    seq1_bytes = hex_to_bytes_list(sequence_list_as_hex[i])
    seq2_bytes = hex_to_bytes_list(sequence_list_as_hex[j])

    alignments = get_full_alignment(seq1_bytes, seq2_bytes)
    best_alignment_s1_s2 = next(alignments)

    aligned1, aligned2 = biopython_alignment_to_lists(best_alignment_s1_s2)

    alignment = [aligned1, aligned2]
    used_indices = [i, j]


    ref_row_idx = 0

    while len(used_indices) < n:
        next_idx = find_next_sequence(distance_score_matrix, used_indices)
        if next_idx is None:
            break

        new_seq = hex_to_bytes_list(sequence_list_as_hex[next_idx])

        old_ref = alignment[ref_row_idx]
        ref_ungapped = [b for b in old_ref if b is not None]

        aln_obj = next(get_full_alignment(ref_ungapped, new_seq))
        new_ref_aligned, new_seq_aligned = biopython_alignment_to_lists(aln_obj)

        alignment = expand_alignment_from_ref(alignment, old_ref, new_ref_aligned)

        alignment[ref_row_idx] = new_ref_aligned

        alignment.append(new_seq_aligned)
        used_indices.append(next_idx)

    return alignment, used_indices



def seperate_client_and_server(df):

    client_df = df[df["src_port"] != 102].copy()
    server_df = df[df["src_port"] == 102].copy()

    return client_df, server_df




def start_sequence_alignment(df):
    df_s7 = filter_s7_packets(df)   #ensure df only contains s7comm packets

    # seperate into client and server session
    client_df, server_df = seperate_client_and_server(df_s7)

    #client
    sequence_list_as_hex_client=client_df["data"].tolist()

    distance_score_matrix_client=get_distance_score_matrix(sequence_list_as_hex_client)
    alignment_client, used_indices_client= build_progressive_alignment(sequence_list_as_hex_client,distance_score_matrix_client)

    #print sequence alignment
    print("Client Alignment")

    show_alignment_block(alignment_client, used_indices_client)

    #server
    sequence_list_as_hex_server = server_df["data"].tolist()


    distance_score_matrix_server = get_distance_score_matrix(sequence_list_as_hex_server)
    alignment_server, used_indices_server =build_progressive_alignment(sequence_list_as_hex_server,distance_score_matrix_server)

    print("Server Alignment")
    show_alignment_block(alignment_server, used_indices_server)


    return alignment_client, alignment_server

