# we must first install cudf and dask-cudf to run this code
import cudf
import cupy as cp
import dask_cudf
import ipaddress
from dask.distributed import Client

# start distributed scheduler (uses all GPUs if available)
client = Client()

# read dataset in GPU chunks (~1 GB per chunk)
df = dask_cudf.read_csv('../../DataSets/electra_s7comm/electra_s7comm.csv', chunksize="1GB")

# convert Time column (vectorized on GPU)
df['Time'] = df['Time'] / 1_000_000

# compute packet size (vectorized)
df['packet_size'] = 4*3 + 1*2 + df['label'].str.len() + 1

# create pair_ip column (still needs CPU fallback since ipaddress not GPU-native)
def create_pair_ip(sip, dip):
    try:
        ip1, ip2 = ipaddress.ip_address(sip), ipaddress.ip_address(dip)
        return f"{min(ip1, ip2)}_{max(ip1, ip2)}"
    except:
        return None

# apply CPU function per partition
df = df.map_partitions(lambda gdf: gdf.assign(pair_ip=gdf.apply_rows(
    create_pair_ip, incols={'sip':'sip', 'dip':'dip'}, outcols={'pair_ip': str}
)))

# split normal vs attacked on GPU
normal = df[df['label'] == 'NORMAL']
attacked = df[df['label'] != 'NORMAL']

# write back to disk in parallel
normal.to_csv('../../DataSets/electra_s7comm/gpu_output/normal/*.csv', single_file=False)
attacked.to_csv('../../DataSets/electra_s7comm/gpu_output/attacked/*.csv', single_file=False)