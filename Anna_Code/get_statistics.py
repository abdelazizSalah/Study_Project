import pandas as pd



def preprocess(df):
    out = df.copy()
    out['app_proto'] = out['app_proto'].replace('undetected:-1:-1', 'none')
    return out


# Task 01 A ###########################################################################################################
#pd.series = column table
#Return count, fraction, percentage for a categorical series
def counts_fractions_helper(series: pd.Series, denom: int):
    counts = series.value_counts()
    fraction = counts / denom
    percentage = (fraction * 100).round(2)
    return pd.DataFrame({'count': counts, 'fraction': fraction, 'percentage': percentage})


def app_layer_packet_distribution(df):
    """
    - How many and which app-layer protocols (excl. 'none')?
    - Fraction of packets per app-layer protocol (incl. 'none').
    - Fraction among ATTACK packets per app-layer protocol (incl. 'none').
    """

    #complete dataset
    total = len(df)
    with_app = df[df['app_proto'] != 'none']

    # presence (excl. 'none')
    num_protocols = with_app['app_proto'].nunique()
    which_protocols = sorted(with_app['app_proto'].unique().tolist())

    # distribution (incl. 'none')
    app_stats = counts_fractions_helper(df['app_proto'], total).sort_values('count', ascending=False)

    # attack-only distribution
    atk = df[df['label_attack'] == 1]
    app_attack_stats = counts_fractions_helper(atk['app_proto'], len(atk)).sort_values('count', ascending=False)

    return {
        'num_app_protocols_excl_none': num_protocols,
        'app_protocol_list_excl_none': which_protocols,
        'app_protocol_stats': app_stats,
        'app_protocol_attack_stats': app_attack_stats,
    }


def transport_layer_packet_distribution(df):
    """
    - Fraction of packets per transport-layer protocol.
    - Fraction among ATTACK packets per transport-layer protocol.
    """

    #complete dataset
    l4_stats = counts_fractions_helper(df['l4_proto'], len(df)).sort_values('count', ascending=False)

    #attack dataset
    atk = df[df['label_attack'] == 1]
    l4_attack_stats = counts_fractions_helper(atk['l4_proto'], len(atk)).sort_values('count', ascending=False)

    return {
        'l4_protocol_stats': l4_stats,
        'l4_protocol_attack_stats': l4_attack_stats,
    }


def transport_and_app_layer_packet_distribution(df):
    """
    - Fraction of packets for each (transport, application) pair vs entire dataset.
    - Fraction among ATTACK packets for each (transport, application) pair.
    """

    #complete dataset
    total = len(df)
    combo_counts = df.groupby(['l4_proto', 'app_proto']).size()

    combo_frac = combo_counts / total
    combo_pct = (combo_frac * 100).round(2)
    combo_stats = (pd.DataFrame({'count': combo_counts,
                                 'fraction': combo_frac,
                                 'percentage': combo_pct})
                   .sort_values('count', ascending=False))

    #attack dataset
    atk = df[df['label_attack'] == 1]
    atk_total = len(atk)
    atk_combo_counts = atk.groupby(['l4_proto', 'app_proto']).size()

    atk_combo_frac = atk_combo_counts / atk_total
    atk_combo_pct = (atk_combo_frac * 100).round(2)
    attack_combo_stats = (pd.DataFrame({'attack_count': atk_combo_counts,
                                        'fraction': atk_combo_frac,
                                        'percentage': atk_combo_pct})
                          .sort_values('attack_count', ascending=False))

    return {
        'combo_stats': combo_stats,
        'attack_combo_stats': attack_combo_stats,
    }


def print_packet_distribution_task1A(df):
    df = preprocess(df) #todo

    app_results = app_layer_packet_distribution(df)
    l4_results = transport_layer_packet_distribution(df)
    combo_results = transport_and_app_layer_packet_distribution(df)

    print("Task 01A")
    print("""
        - How many and which app-layer protocols (excl. 'none')?
        - Fraction of packets per app-layer protocol (incl. 'none').
        - Fraction among ATTACK packets per app-layer protocol (incl. 'none').
        """)
    print(f"Number of distinct app-layer protocols (excl. 'none'): "
          f"{app_results['num_app_protocols_excl_none']}")
    print(f"Protocols: {', '.join(app_results['app_protocol_list_excl_none'])}\n")
    print("Complete Dataset:")
    print(app_results['app_protocol_stats'].to_string())
    print("Attack Packets:")
    print(app_results['app_protocol_attack_stats'].to_string())

    print("""
    - Fraction of packets per transport-layer protocol.
    - Fraction among ATTACK packets per transport-layer protocol.
    """)
    print("Complete Dataset:")
    print(l4_results['l4_protocol_stats'].to_string())
    print("Attack Packets:")
    print(l4_results['l4_protocol_attack_stats'].to_string())

    print("""
    - Fraction of packets for each (transport, application) pair vs entire dataset.
    - Fraction among ATTACK packets for each (transport, application) pair.
    """)
    print("Complete Dataset:")
    print(combo_results['combo_stats'].to_string())
    print("Attack Packets:")
    print(combo_results['attack_combo_stats'].to_string())

    return
# Task 01 B ###########################################################################################################

def agg_frame_len(sub_df):
    if sub_df.empty:
        return pd.DataFrame(columns=['mean_len', 'std_len', 'median_len'])
    return sub_df.groupby('app_proto')['frame_len'].agg(
        mean_len='mean', # built-in pandas/NumPy function for average
        std_len='std',  # built-in for standard deviation
        median_len='median' # built-in for median (middle value)
    ).round(2)


def packet_length_stats(df):
    """
    Only consider packets with application data.
    Computes average, std, and median of frame_len (packet size in bytes) for each protocol type
    for control dataset subset and attack packets subset only.
    """

    df = df[df['app_payload_len'] > 0]   # only packets carrying application data

    # --- Split into attack / normal subsets ---
    normal_df = df[df['label_attack'] == 0]
    attack_df = df[df['label_attack'] == 1]

    # --- Define helper for aggregation ---

    # --- Compute statistics for both sets ---
    normal_stats = agg_frame_len(normal_df)
    attack_stats = agg_frame_len(attack_df)

    return {'normal': normal_stats, 'attack': attack_stats}



#pandas aggregation functions automatically exclude NaNs
#NaN for first packet (that was 0 before)
def agg_iat_per_pair(sub_df):
    if sub_df.empty:
        return pd.DataFrame(columns=['mean_iat', 'std_iat', 'median_iat'])
    return sub_df.groupby('pair_id')['iat_pair'].agg(
            mean_iat='mean',  # average inter-arrival time per pair
            std_iat='std',  # variation within that pair’s traffic
            median_iat='median'  # middle timing value if results sorted
        ).round(6)


#todo currently it computes average for EACH HOST PAIR
def iat_per_pair_stats(df):
    #todo consider that the iat entry for the first packet is 0!
    """
    Computes average, average, std, and median inter-arrival time between same host pair for control dataset and attack packets subset only.
    """

    normal = df[df['label_attack'] == 0]
    attack = df[df['label_attack'] == 1]

    results = []

    # --- NORMAL packets ---
    normal_stats = agg_iat_per_pair(normal)
    attack_stats = agg_iat_per_pair(attack)

    return {'normal': normal_stats, 'attack': attack_stats}



def print_packet_length_distribution_and_iat_task1B(df):
    df = preprocess(df) #todo consider where to put
    print("Task 01B")
    print("""
    Only consider packets with application data.
    Computes mean, std, and median of frame_len (packet size in bytes) for each protocol type
    for control dataset and attack packets subset only.
    """)
    packet_len_results = packet_length_stats(df)
    print("\nControl Dataset:")

    print(packet_len_results['normal'].to_string())
    print("\nAttack Dataset:")
    print(packet_len_results['attack'].to_string())

    print("""
    Computes average, average, std, and median inter-arrival time between same host pair for control dataset and attack packets subset only.
    """)
    iat_stats=iat_per_pair_stats(df)
    print("\nControl Dataset:")
    print(iat_stats["normal"].head(20).to_string())

    print("\nAttack Dataset:")
    print(iat_stats["attack"].head(20).to_string())

    return 0



# Task 01 C ###########################################################################################################

def count_host_pairs(df):
    """How many pairs of hosts for complete and attack dataset."""
    normal = df[df['label_attack'] == 0]
    attack = df[df['label_attack'] == 1]

    return {
        'complete': df['pair_id'].nunique(),
        'normal': normal['pair_id'].nunique(),
        'attack': attack['pair_id'].nunique()
    }


def agg_iat_per_pair_per_app_proto(sub_df):
    if sub_df.empty:
        return pd.DataFrame(columns=['mean_iat', 'std_iat', 'median_iat'])
    return (
        sub_df
        .groupby(['pair_id', 'app_proto'])['iat_pair']
        .agg(mean_iat='mean', std_iat='std', median_iat='median')
        .round(6)
    )


#iat not only grouped by host pair but also by app protocol
def iat_per_pair_per_proto_stats(df):
    """
    Computes average, average, std, and median inter-arrival time between same host pair from SAME APPLICATION LAYER PROTOCOL
    for control dataset and attack packets subset only.
    """
    normal = df[df['label_attack'] == 0]
    attack = df[df['label_attack'] == 1]

    results = []

    # --- NORMAL packets ---
    normal_stats = agg_iat_per_pair_per_app_proto(normal)
    attack_stats = agg_iat_per_pair_per_app_proto(attack)

    return {'normal': normal_stats, 'attack': attack_stats}


def print_packet_distribution_task1C(df):
    df=preprocess(df)
    print("""How many pairs of hosts for complete and attack dataset.""")
    print(count_host_pairs(df))
    print("""
     Computes average, average, std, and median inter-arrival time between same host pair from SAME APPLICATION LAYER PROTOCOL
     for control dataset and attack packets subset only.
     """)
    iat_proto_results = iat_per_pair_per_proto_stats(df)
    print("\nControl Dataset:")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(iat_proto_results['normal'].to_string())
        print("\nAttack Dataset:")
        print(iat_proto_results['attack'].to_string())

    return