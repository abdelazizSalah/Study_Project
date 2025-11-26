import numpy as np


def detect_unit_fields(aligned_msgs):
    """
    aligned_msgs: NumPy-Array shape (n_msgs, msg_len), dtype=object (Bytes oder None)

    Rückgabe:
        Liste von dicts mit:
        {
            "start": spaltenindex,
            "end":   spaltenindex (am Anfang immer == start),
            "is_static": True/False
        }
    """
    unit_fields = []

    for col in range(aligned_msgs.shape[1]):
        col_vals = aligned_msgs[:, col]

        # Wenn irgendwo ein Gap ist -> dynamic
        if any(v is None for v in col_vals):
            is_static = False
        else:
            unique_vals = np.unique(col_vals)
            is_static = (len(unique_vals) == 1)

        unit_fields.append({
            "start": col,
            "end": col,
            "is_static": is_static,
        })

    return unit_fields




def merge_static_fields(unit_fields):
    """
    unit_fields: Liste von dicts wie aus detect_unit_fields()

    Merge nur aufeinanderfolgende statische Felder.
    Dynamische bleiben Einzelfelder.

    Rückgabe: Liste von dicts:
        {
            "start": start_index,
            "end": end_index,
            "is_static": True/False
        }
    """
    merged = []
    i = 0
    n = len(unit_fields)

    while i < n:
        uf = unit_fields[i]

        if uf["is_static"]:
            start = uf["start"]
            end = uf["end"]

            j = i + 1
            while (
                j < n
                and unit_fields[j]["is_static"]
                and unit_fields[j]["start"] == end + 1
            ):
                end = unit_fields[j]["end"]
                j += 1

            merged.append({
                "start": start,
                "end": end,
                "is_static": True,
            })
            i = j
        else:
            # dynamisches Feld bleibt wie es ist
            merged.append({
                "start": uf["start"],
                "end": uf["end"],
                "is_static": False,
            })
            i += 1

    return merged


def build_keyword_candidates(merged_fields):
    """
    merged_fields: Liste von dicts mit start/end/is_static

    Rückgabe: Liste von dicts mit zusätzlicher field_id
    """
    keyword_candidates = []
    for i, f in enumerate(merged_fields):
        kc = {
            "field_id": i,
            "start": f["start"],
            "end": f["end"],
            "is_static": f["is_static"],
        }
        keyword_candidates.append(kc)

    return keyword_candidates



def build_fields_and_candidates_from_alignment(alignment):
    """
    alignment: List[List[int | None]]

    Rückgabe:
        unit_fields:   Liste von dicts (start=end=col, is_static)
        merged_fields: Liste von dicts (gemergte statische Blöcke)
        keyword_candidates: Liste von dicts (wie merged_fields, plus field_id)
    """
    aligned_np = np.array(alignment, dtype=object)

    unit_fields = detect_unit_fields(aligned_np)
    merged_fields = merge_static_fields(unit_fields)
    keyword_candidates = build_keyword_candidates(merged_fields)

    return unit_fields, merged_fields, keyword_candidates