#!/usr/bin/env python3
"""
  python3 task3_ngram_detector.py \
    --train-pcaps run1.pcap run2.pcap run8.pcap \
    --val-pcap send_a_fake_command_modbus_6RTU_with_operate.pcap \
    --val-labels send_a_fake_command_modbus_6RTU_with_operate_labeled.csv \
    --test-pcap characterization_modbus_6RTU_with_operate.pcap \
    --test-labels characterization_modbus_6RTU_with_operate_labeled.csv \
    --ngrams 2 3 4 \
    --bf-capacity 10000000 --bf-fp 0.01 \
    --outdir task3_out --force-threshold 0.05
"""

import argparse
import csv
import math
import sys
import os
import hashlib
import pyshark
from datetime import datetime
from typing import Iterable, Sequence, List, Dict, Optional, Tuple

# Bloom filter

class BloomFilter:
    def __init__(self, capacity: int, fp_rate: float):
        if capacity <= 0 or not (0 < fp_rate < 1):
            raise ValueError("Invalid capacity/fp_rate")
        ln2 = math.log(2)
        self.m = int(-capacity * math.log(fp_rate) / (ln2 ** 2))  # total bits
        self.k = max(1, int(round((self.m / capacity) * ln2)))    # # of hashes
        self.bits = bytearray((self.m + 7) // 8)

    def _set_bit(self, i: int): self.bits[i // 8] |= (1 << (i % 8))
    def _get_bit(self, i: int) -> bool: return bool(self.bits[i // 8] & (1 << (i % 8)))

    def _hashes(self, data: bytes) -> Iterable[int]:
        h1 = int.from_bytes(hashlib.sha1(data).digest()[:8], "big")
        h2 = int.from_bytes(hashlib.sha256(data).digest()[:8], "big") | 1
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, data: bytes):
        for idx in self._hashes(data):
            self._set_bit(idx)

    def __contains__(self, data: bytes) -> bool:
        for idx in self._hashes(data):
            if not self._get_bit(idx):
                return False
        return True


# Helpers

def load_labels(path: Optional[str]) -> Dict[int, int]:
    if not path:
        return {}
    labels = {}
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f, delimiter=';')
        for row in rdr:
            if len(row) != 2:
                continue
            labels[int(row[0])] = int(row[1])
    return labels


def ngrams_from_bytes(b: bytes, n: int) -> Iterable[bytes]:
    if n <= 0: return []
    L = len(b)
    for i in range(0, max(0, L - n + 1)):
        yield b[i:i+n]


def mixed_ngrams(b: bytes, ns: Sequence[int]) -> List[bytes]:
    grams = []
    for n in ns:
        grams.extend(ngrams_from_bytes(b, n))
    return grams


def packet_bytes(pkt) -> Optional[bytes]:
    """Extract raw bytes from a pyshark packet."""
    try:
        if hasattr(pkt, "frame_raw"):
            val = getattr(pkt, "frame_raw")
            hexstr = getattr(val, "value", None) or str(val)
            hexstr = hexstr.strip().replace(":", "").replace(" ", "")
            return bytes.fromhex(hexstr)
    except Exception:
        pass
    try:
        raw = pkt.get_raw_packet()
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
    except Exception:
        pass
    return None



# Core logic

def train_bloom(train_pcaps: Sequence[str],
                ns: Sequence[int],
                bf_capacity: int,
                bf_fp: float,
                log) -> BloomFilter:
    bf = BloomFilter(bf_capacity, bf_fp)
    for pcap in train_pcaps:
        log(f"[train] Reading {pcap} ...")
        cap = pyshark.FileCapture(pcap, use_json=True, include_raw=True)
        try:
            for pkt in cap:
                b = packet_bytes(pkt)
                if not b:
                    continue
                seen = set()
                for n in ns:
                    for g in ngrams_from_bytes(b, n):
                        seen.add(g)
                for g in seen:
                    bf.add(g)
        finally:
            cap.close()
    return bf


def score_packets(pcap: str, ns: Sequence[int], bf: BloomFilter):
    cap = pyshark.FileCapture(pcap, use_json=True, include_raw=True)
    try:
        for pkt in cap:
            try:
                fn = int(pkt.number)
            except Exception:
                fn = -1
            b = packet_bytes(pkt)
            if not b:
                yield (fn, 0.0, 0, 0)
                continue
            grams = mixed_ngrams(b, ns)
            T = len(grams)
            if T == 0:
                yield (fn, 0.0, 0, 0)
                continue
            newN = sum(1 for g in grams if g not in bf)
            yield (fn, newN / T, T, newN)
    finally:
        cap.close()


def grid_search_threshold(val_scores: List[Tuple[int, float]]):
    if not val_scores:
        return 0.5, {"note": "no validation data"}
    scores = sorted({s for _, s in val_scores})
    candidates = [0.0] + [(scores[i]+scores[i+1])/2 for i in range(len(scores)-1)] + [1.0]

    def f1_at(th):
        tp=fp=tn=fn=0
        for y,s in val_scores:
            yhat = 1 if s > th else 0
            if y==1 and yhat==1: tp+=1
            elif y==0 and yhat==1: fp+=1
            elif y==0 and yhat==0: tn+=1
            elif y==1 and yhat==0: fn+=1
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1  = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        return f1, {"precision":prec,"recall":rec,"f1":f1,"tp":tp,"fp":fp,"tn":tn,"fn":fn}

    best_th, best_f1, best_m = 0.5, -1.0, {}
    for th in candidates:
        f1, m = f1_at(th)
        if f1 > best_f1:
            best_th, best_f1, best_m = th, f1, m
    best_m["best_threshold"] = best_th
    return best_th, best_m


# CLI

def main():
    ap = argparse.ArgumentParser(description="Task 3: N-gram anomaly detector using Bloom filter")
    ap.add_argument("--train-pcaps", nargs="+", required=True)
    ap.add_argument("--val-pcap")
    ap.add_argument("--val-labels")
    ap.add_argument("--test-pcap", required=True)
    ap.add_argument("--test-labels")
    ap.add_argument("--ngrams", nargs="+", type=int, default=[2,3,4])
    ap.add_argument("--bf-capacity", type=int, default=10_000_000)
    ap.add_argument("--bf-fp", type=float, default=0.01)
    # New output controls
    ap.add_argument("--outdir", default="task3_out")
    ap.add_argument("--results", default=None, help="Per-packet CSV path (optional)")
    ap.add_argument("--log", default=None, help="Log file path (optional)")
    # Force threshold option
    ap.add_argument("--force-threshold", type=float, default=None,
                    help="If set, skip validation threshold search and use this value (e.g. 0.05)")
    args = ap.parse_args()

    # Prepare n values
    ns=[n for n in args.ngrams if n>1]

    # Prepare output paths
    os.makedirs(args.outdir, exist_ok=True)
    testbase = os.path.splitext(os.path.basename(args.test_pcap))[0]
    results_path = args.results or os.path.join(args.outdir, f"per_packet_{testbase}.csv")
    if args.log:
        log_path = args.log
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.outdir, f"run_{ts}.log")

    # Simple logger that writes to file + stderr
    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    def log(msg: str):
        line = msg.rstrip()
        print(line, file=sys.stderr)
        print(line, file=log_fh)

    log(f"[setup] outdir={args.outdir}")
    log(f"[setup] results={results_path}")
    log(f"[setup] log={log_path}")
    log(f"[setup] ngrams={ns}  bf-capacity={args.bf_capacity}  bf-fp={args.bf_fp}")

    # 1. Train
    bf = train_bloom(args.train_pcaps, ns, args.bf_capacity, args.bf_fp, log)
    log(f"[train] Bloom filter: bits={bf.m}, hashes={bf.k}")

    # 2. Validation threshold search (unless --force-threshold is provided)
    best_th = 0.5
    if args.force_threshold is not None:
        best_th = args.force_threshold
        log(f"[val] force-threshold provided: using τ={best_th:.6f} (skipping validation search)")
    else:
        if args.val_pcap and args.val_labels:
            vlabels = load_labels(args.val_labels)
            val_scores=[]
            log(f"[val] Scoring validation PCAP: {args.val_pcap}")
            for fn,sc,T,newN in score_packets(args.val_pcap,ns,bf):
                if fn in vlabels:
                    val_scores.append((vlabels[fn],sc))
            best_th,metrics=grid_search_threshold(val_scores)
            log(f"[val] Best threshold={best_th:.6f} "
                f"F1={metrics.get('f1',0):.3f} "
                f"P={metrics.get('precision',0):.3f} "
                f"R={metrics.get('recall',0):.3f} "
                f"TP={metrics.get('tp',0)} FP={metrics.get('fp',0)} "
                f"TN={metrics.get('tn',0)} FN={metrics.get('fn',0)}")
        else:
            log("[val] No validation provided; using default τ=0.5")

    # 3. Test phase (save per-packet CSV)
    tlabels = load_labels(args.test_labels) if args.test_labels else {}
    have_labels=bool(tlabels)
    with open(results_path, "w", newline="", encoding="utf-8") as outcsv:
        header = ["frame_number","score","decision"]
        if have_labels: header.append("label")
        outcsv.write("# " + ";".join(header) + "\n")

        tp=fp=tn=fn=0
        log(f"[test] Scoring test PCAP: {args.test_pcap}")
        for fn,sc,T,newN in score_packets(args.test_pcap,ns,bf):
            yhat = 1 if sc>best_th else 0
            if have_labels and fn in tlabels:
                y=tlabels[fn]
                if y==1 and yhat==1: tp+=1
                elif y==0 and yhat==1: fp+=1
                elif y==0 and yhat==0: tn+=1
                elif y==1 and yhat==0: fn+=1
                row = f"{fn};{sc:.6f};{'attack' if yhat else 'normal'};{y}\n"
            else:
                row = f"{fn};{sc:.6f};{'attack' if yhat else 'normal'}\n"
            outcsv.write(row)

    if have_labels:
        prec=tp/(tp+fp) if tp+fp else 0
        rec =tp/(tp+fn) if tp+fn else 0
        f1  =2*prec*rec/(prec+rec) if prec+rec else 0
        log(f"[test] METRICS  threshold={best_th:.6f} "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} TP={tp} FP={fp} TN={tn} FN={fn}")
    else:
        log(f"[test] Completed (unlabeled). threshold={best_th:.6f}")

    log(f"[done] Per-packet CSV saved to: {results_path}")
    log(f"[done] Log saved to: {log_path}")
    log_fh.close()

if __name__=="__main__":
    main()

