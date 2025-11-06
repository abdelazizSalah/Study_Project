
(rapids_env) haiti:~/Study_Project/Abdelaziz_Codes/Sheet1_codes> python task2_script.py
## start task2 processing
### processing task 2a Q
Loaded 161 files.
Total time: 2.51 seconds
before sorting
after sorting
data head before flows creation                                                        timestamp src_ip  ... direction        iat
app_proto     pair_id                                                    ...
DHCPv6 Client 08:00:27:2e:cf:f4__33:33:00:01:00:02 0   18.047408    NaN  ...         1        NaN
                                                   1   18.471999    NaN  ...         1   0.424591
                                                   2   18.472055    NaN  ...         1   0.000056
                                                   3  114.593292    NaN  ...         1  96.121237
                                                   4  115.593963    NaN  ...         1   1.000671

[5 rows x 6 columns]
Index(['timestamp', 'src_ip', 'dst_ip', 'frame_len', 'direction', 'iat'], dtype='object')
2
4
6
 data from attacked
 flows_2min len: 1183948,  flows_4min len: 1183948,  flows_6min len: 1183948.
attacked number of flows per protocol
 2min: s7comm           1159040
unknown            22563
NTP                 1593
DHCPv6 Client        505
llmnr                 80
NBDS                  78
NBNS                  69
mDNS                  20
Name: count, dtype: int64
 4min: s7comm           1159040
unknown            22563
NTP                 1593
DHCPv6 Client        505
llmnr                 80
NBDS                  78
NBNS                  69
mDNS                  20
Name: count, dtype: int64
 6min: s7comm           1159040
unknown            22563
NTP                 1593
DHCPv6 Client        505
llmnr                 80
NBDS                  78
NBNS                  69
mDNS                  20
Name: count, dtype: int64
Loaded 3 files.
Total time: 2.16 seconds
before sorting

after sorting
data head before flows creation                                                        timestamp src_ip  ... direction           iat
app_proto     pair_id                                                    ...                     
DHCPv6 Client 08:00:27:2e:cf:f4__33:33:00:01:00:02 0  349.503720    NaN  ...         1           NaN
                                                   1  349.503721    NaN  ...         1  9.536743e-07
                                                   2  349.503847    NaN  ...         1  1.258850e-04
                                                   3  350.504093    NaN  ...         1  1.000246e+00
                                                   4  350.504167    NaN  ...         1  7.414818e-05

[5 rows x 6 columns]
Index(['timestamp', 'src_ip', 'dst_ip', 'frame_len', 'direction', 'iat'], dtype='object')
2
4
6
 data from normal
 flows_2min len: 128522,  flows_4min len: 128522,  flows_6min len: 128522.
normal number of flows per protocol
 2min: s7comm           93230
unknown          29710
NTP               4629
DHCPv6 Client      602
llmnr              132
NBNS               126
NBDS                93
Name: count, dtype: int64
 4min: s7comm           93230
unknown          29710
NTP               4629
DHCPv6 Client      602
llmnr              132
NBNS               126
NBDS                93
Name: count, dtype: int64
 6min: s7comm           93230
unknown          29710
NTP               4629
DHCPv6 Client      602
llmnr              132
NBNS               126
NBDS                93
Name: count, dtype: int64
### processing task 2a E
task2a starting
loading dataset in 0.034946441650390625 secs
      Time          sip          dip  ... packet_size           label request
0  0.00033  10.70.38.54  10.70.38.52  ...          21  MITM_UNALTERED       1
1  0.00033  10.70.38.54  10.70.38.52  ...       46553  MITM_UNALTERED       1
2  0.00033  10.70.38.54  10.70.38.52  ...        2368  MITM_UNALTERED       1
3  0.00033  10.70.38.54  10.70.38.52  ...           1  MITM_UNALTERED       1
4  0.00033  10.70.38.54  10.70.38.52  ...           1  MITM_UNALTERED       1

[5 rows x 7 columns]
Index(['timestamp', 'sip', 'dip', 'pair_ip', 'frame_len', 'label', 'request'], dtype='object')
after selecting the features
   timestamp          sip          dip  frame_len                  pair_ip
0    0.00033  10.70.38.54  10.70.38.52         21  10.70.38.52_10.70.38.54
1    0.00033  10.70.38.54  10.70.38.52      46553  10.70.38.52_10.70.38.54
2    0.00033  10.70.38.54  10.70.38.52       2368  10.70.38.52_10.70.38.54
3    0.00033  10.70.38.54  10.70.38.52          1  10.70.38.52_10.70.38.54
4    0.00033  10.70.38.54  10.70.38.52          1  10.70.38.52_10.70.38.54
before grouping
after the direction
after timestamp conversion
adding iat
starting creating flows 2 mins
writing csv file attacked0.csv
processing time to write to the disk: 0.02672433853149414
data head before flows creation                            timestamp          sip          dip  frame_len  direction  iat
pair_ip
10.70.38.52_10.70.38.54 0    0.00033  10.70.38.54  10.70.38.52          1          1  NaN
                        1    0.00033  10.70.38.54  10.70.38.52          1          1  0.0
                        2    0.00033  10.70.38.54  10.70.38.52          2          1  0.0
                        3    0.00033  10.70.38.54  10.70.38.52          1          1  0.0
                        4    0.00033  10.70.38.54  10.70.38.52          3          1  0.0
Index(['timestamp', 'sip', 'dip', 'frame_len', 'direction', 'iat'], dtype='object')
 Creating 2-minute windows...
max timestamp: 10.257091
 Completed 2-minute flow creation using 128 cores.
starting creating flows 4 mins
 Creating 4-minute windows...
max timestamp: 10.257091
 Completed 4-minute flow creation using 128 cores.
starting creating flows 6 mins
 Creating 6-minute windows...
max timestamp: 10.257091
 Completed 6-minute flow creation using 128 cores.
 data from attacked
 flows_2min len: 50000,  flows_4min len: 50000,  flows_6min len: 50000.
attacked number of flows per protocol
 2min: 10.70.38.52_10.70.38.54    50000
Name: count, dtype: int64
 4min: 10.70.38.52_10.70.38.54    50000
Name: count, dtype: int64
 6min: 10.70.38.52_10.70.38.54    50000
Name: count, dtype: int64
task2a starting
loading dataset in 0.03131508827209473 secs
   Time          sip          dip                  pair_ip  packet_size   label  request
0   0.0  10.70.38.53  10.70.38.52  10.70.38.52_10.70.38.53            1  NORMAL        1
1   0.0  10.70.38.53  10.70.38.52  10.70.38.52_10.70.38.53            1  NORMAL        1
2   0.0  10.70.38.53  10.70.38.52  10.70.38.52_10.70.38.53            2  NORMAL        1
3   0.0  10.70.38.53  10.70.38.52  10.70.38.52_10.70.38.53            1  NORMAL        1
4   0.0  10.70.38.53  10.70.38.52  10.70.38.52_10.70.38.53            2  NORMAL        1
Index(['timestamp', 'sip', 'dip', 'pair_ip', 'frame_len', 'label', 'request'], dtype='object')
after selecting the features
     timestamp          sip          dip  frame_len                  pair_ip
0     0.000000  10.70.38.53  10.70.38.52          1  10.70.38.52_10.70.38.53
1     0.000000  10.70.38.53  10.70.38.52          1  10.70.38.52_10.70.38.53
2     0.000000  10.70.38.53  10.70.38.52          2  10.70.38.52_10.70.38.53
3     0.000000  10.70.38.53  10.70.38.52          1  10.70.38.52_10.70.38.53
4     0.000000  10.70.38.53  10.70.38.52          2  10.70.38.52_10.70.38.53
372   0.048616  10.70.38.51  10.70.38.52          1  10.70.38.51_10.70.38.52
373   0.048616  10.70.38.51  10.70.38.52          1  10.70.38.51_10.70.38.52
374   0.048616  10.70.38.51  10.70.38.52         16  10.70.38.51_10.70.38.52
375   0.048616  10.70.38.51  10.70.38.52         12  10.70.38.51_10.70.38.52
376   0.048616  10.70.38.51  10.70.38.52         16  10.70.38.51_10.70.38.52
before grouping
after the direction
after timestamp conversion
adding iat
starting creating flows 2 mins
writing csv file normal0.csv
processing time to write to the disk: 0.022260427474975586
data head before flows creation                            timestamp          sip          dip  frame_len  direction  iat
pair_ip
10.70.38.51_10.70.38.52 0   0.048616  10.70.38.51  10.70.38.52          1          1  NaN
                        1   0.048616  10.70.38.51  10.70.38.52          1          1  0.0
                        2   0.048616  10.70.38.51  10.70.38.52         16          1  0.0
                        3   0.048616  10.70.38.51  10.70.38.52          1          1  0.0
                        4   0.048616  10.70.38.51  10.70.38.52         12          1  0.0
Index(['timestamp', 'sip', 'dip', 'frame_len', 'direction', 'iat'], dtype='object')
 Creating 2-minute windows...
max timestamp: 9.7458
 Completed 2-minute flow creation using 128 cores.
starting creating flows 4 mins
 Creating 4-minute windows...
max timestamp: 9.7458
 Completed 4-minute flow creation using 128 cores.
starting creating flows 6 mins
 Creating 6-minute windows...
max timestamp: 9.7458
 Completed 6-minute flow creation using 128 cores.
 data from normal
 flows_2min len: 50000,  flows_4min len: 50000,  flows_6min len: 50000.
normal number of flows per protocol
 2min: 10.70.38.52_10.70.38.53    37710
10.70.38.51_10.70.38.52    12290
Name: count, dtype: int64
 4min: 10.70.38.52_10.70.38.53    37710
10.70.38.51_10.70.38.52    12290
Name: count, dtype: int64
 6min: 10.70.38.52_10.70.38.53    37710
10.70.38.51_10.70.38.52    12290
Name: count, dtype: int64
### processing task 2b Q
start processing task2b
start processing task2b
### processing task 2b E
start processing task2b
start processing task2b
### processing task 2c Q
start processing task2c
1688.5687351226807 3002.425534248352
1688.5687351226807 3002.425534248352
1688.5687351226807 3002.425534248352
213 424
213 424
213 424
start processing task2c
8281.823318958286 13752.347853183746
8281.823318958286 13752.347853183746
8281.823318958286 13752.347853183746
128 241
128 241
128 241
### processing task 2c E
start processing task2c
0.9909059999999998 1.0104359999999994
0.9909059999999998 1.0104359999999994
0.9909059999999998 1.0104359999999994
46563 64274
46563 64274
46563 64274
start processing task2c
0.7049880000000011 1.0040810000000002
0.7049880000000011 1.0040810000000002
0.7049880000000011 1.0040810000000002
1065353216 1123680256
1065353216 1123680256
1065353216 1123680256
### processing task 2d Q
Skipping flow 0: no valid embeddings found.
Skipping flow 1: no valid embeddings found.
Skipping flow 2: no valid embeddings found.
Best KL divergences per flow: []
Skipping flow 0: no valid embeddings found.
Skipping flow 1: no valid embeddings found.
Skipping flow 2: no valid embeddings found.
Best KL divergences per flow: []
### processing task 2d E
Skipping flow 0: no valid embeddings found.
Skipping flow 1: no valid embeddings found.
Skipping flow 2: no valid embeddings found.
Best KL divergences per flow: []
Skipping flow 0: no valid embeddings found.
Skipping flow 1: no valid embeddings found.
Skipping flow 2: no valid embeddings found.
Best KL divergences per flow: []
### processing task 2e Q
generating bytes
the pcap file to be processed ['../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/master.pcap', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/attacker.pcap', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/hmi.pcap']
time taken to load all list all files 0.0004420280456542969
max workers: 128
 now creating packet list...
 the given path is ../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/master.pcap
 the given path is ../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/attacker.pcap
 the given path is ../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set/hmi.pcap
packet len: 421080
[b"\x01\x80\xc2\x00\x00\x0e\x08\x00'.\xcf\xf4\x88\xcc\x02\r\x07windows-7-vm\x04\t\x07port-001\x06\x02\x00\x14\n\x0cWINDOWS-7-VM\x0c#innotek GmbH VirtualBox,1.2,0 + HMI\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\n\n\n\x14\x02\x00\x00\x00\x01\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x08\x00'.\xcf\xf4\xfe\t\x00\x12\x0f\x01\x03\xec\x03\x00\x1e\x00\x00", b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\x0c\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0b\xc0\x11\x00f\n\n\n\x0b\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\r\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x01\x80\xc2\x00\x00\x0e\x00\x1b\x1b44\xce\x88\xcc\x02\x07\x04\x00\x1b\x1b44\xcb\x04\t\x07port-003\x06\x02\x00\x14\x0c2Siemens AG SIMATIC S7 MULTIPORT SCALANCE X 200 FAM\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\x00\x00\x00\x00\x02\x00\x00\x00\x03\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x00\x1b\x1b44\xcb\xfe\t\x00\x12\x0f\x01\x03l\x00\x00\x0f\x00\x00']
packet len: 28522
[b"\x01\x80\xc2\x00\x00\x0e\x08\x00'.\xcf\xf4\x88\xcc\x02\r\x07windows-7-vm\x04\t\x07port-001\x06\x02\x00\x14\n\x0cWINDOWS-7-VM\x0c#innotek GmbH VirtualBox,1.2,0 + HMI\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\n\n\n\x14\x02\x00\x00\x00\x01\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x08\x00'.\xcf\xf4\xfe\t\x00\x12\x0f\x01\x03\xec\x03\x00\x1e\x00\x00", b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\x0c\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0b\xc0\x11\x00f\n\n\n\x0b\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\r\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x01\x80\xc2\x00\x00\x0e\x00\x1b\x1b44\xce\x88\xcc\x02\x07\x04\x00\x1b\x1b44\xcb\x04\t\x07port-003\x06\x02\x00\x14\x0c2Siemens AG SIMATIC S7 MULTIPORT SCALANCE X 200 FAM\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\x00\x00\x00\x00\x02\x00\x00\x00\x03\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x00\x1b\x1b44\xcb\xfe\t\x00\x12\x0f\x01\x03l\x00\x00\x0f\x00\x00']
packet len: 421080
[b"\x01\x80\xc2\x00\x00\x0e\x08\x00'.\xcf\xf4\x88\xcc\x02\r\x07windows-7-vm\x04\t\x07port-001\x06\x02\x00\x14\n\x0cWINDOWS-7-VM\x0c#innotek GmbH VirtualBox,1.2,0 + HMI\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\n\n\n\x14\x02\x00\x00\x00\x01\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x08\x00'.\xcf\xf4\xfe\t\x00\x12\x0f\x01\x03\xec\x03\x00\x1e\x00\x00", b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\x0c\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0b\xc0\x11\x00f\n\n\n\x0b\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\xff\xff\xff\xff\xff\xff\x00\x1b\x1b\x17\xf8\x82\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x1b\x1b\x17\xf8\x82\n\n\n\n\n\x0c\xc0\x10\x00f\n\n\n\r\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x01\x80\xc2\x00\x00\x0e\x00\x1b\x1b44\xce\x88\xcc\x02\x07\x04\x00\x1b\x1b44\xcb\x04\t\x07port-003\x06\x02\x00\x14\x0c2Siemens AG SIMATIC S7 MULTIPORT SCALANCE X 200 FAM\x0e\x04\x00\x80\x00\x80\x10\x14\x05\x01\x00\x00\x00\x00\x02\x00\x00\x00\x03\x08+\x06\x01\x04\x01\x81\xc0n\xfe\x08\x00\x0e\xcf\x02\x00\x00\x00\x00\xfe\n\x00\x0e\xcf\x05\x00\x1b\x1b44\xcb\xfe\t\x00\x12\x0f\x01\x03l\x00\x00\x0f\x00\x00']
time taken to process all files 218.79646229743958
time taken to conver and write all files 0.49498438835144043
 now creating pairs
421080
28522
421080
Killed
