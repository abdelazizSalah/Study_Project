How to run the scripts.
-the path is saved into the python files.



and the run1_12rtu_fixed.pcap is the run1_12rtu with -4  corrupted packets.

Library used: 

argparse
os
collections
Counter 
defaultdict
import pandas as pd
import pyshark

import argparse
import csv
import math
import sys
import os
import hashlib
import pyshark
from datetime import datetime
from typing import Iterable, Sequence, List, Dict, Optional, Tupl