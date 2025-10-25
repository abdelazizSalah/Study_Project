import pandas as pd
import pyshark 
import numpy as np

hmi_pcap_file_path= '../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/hmi.pcap/hmi.pcap'
master_pcap_file_path= '../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/master.pcap/master.pcap'
attacker_pcap_file_path= '../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/attacker.pcap/attacker.pcap'


hmi_pcap = pyshark.FileCapture(hmi_pcap_file_path, only_summaries=True,keep_packets=False)
master_pcap = pyshark.FileCapture(master_pcap_file_path, only_summaries=True,keep_packets=False)
attacker_pcap = pyshark.FileCapture(attacker_pcap_file_path, only_summaries=True,keep_packets=False)

# creating dataframe 
hmi_packets = []
for packet in hmi_pcap:
    hmi_packets.append({
        "timestamp": float(packet.time),
        "source": packet.source,
        "destination": packet.destination,
        "protocol": packet.protocol,
        "length": int(packet.length)
    })

hmi_pcap.close()
df_hmi = pd.DataFrame(hmi_packets)
print("HMI DataFrame shape: ", df_hmi.shape)
print(df_hmi.head())

# this scripts takes forever to run
