import pandas as pd
import ipaddress
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

input_file = '../../DataSets/electra_s7comm/electra_s7comm.csv'
output_dir = Path('../../DataSets/electra_s7comm/output')
(output_dir / 'normal').mkdir(parents=True, exist_ok=True)
(output_dir / 'attacked').mkdir(parents=True, exist_ok=True)

def compute_packet_size(row):
    return 4*3 + 1*2 + len(row['label']) + 1

def create_pair_ip(row):
    try:
        ip1 = ipaddress.ip_address(row["sip"])
        ip2 = ipaddress.ip_address(row["dip"])
        return f"{min(ip1, ip2)}_{max(ip1, ip2)}"
    except:
        return None

def process_chunk(chunk_id, chunk):
    chunk.Time = chunk.Time / 1_000_000
    chunk["packet_size"] = chunk.apply(compute_packet_size, axis=1)
    chunk["pair_ip"] = chunk.apply(create_pair_ip, axis=1)
    chunk_final = chunk[["Time","sip","dip","pair_ip","packet_size","label","request"]].sort_values("Time")

    normal = chunk_final[chunk_final['label']=="NORMAL"]
    attacked = chunk_final[chunk_final['label']!="NORMAL"]

    normal.to_csv(output_dir / f'normal/normal_data_{chunk_id}.csv', index=False)
    attacked.to_csv(output_dir / f'attacked/attacked_data_{chunk_id}.csv', index=False)
    return f"Chunk {chunk_id} done"

def main():
    chunks = pd.read_csv(input_file, chunksize=1_000_000)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
        for f in as_completed(futures):
            print(f.result())

if __name__ == "__main__":
    main()

## Parallelized processing of large CSV file into normal and attacked datasets based on labels.