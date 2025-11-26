import numpy as np


def detect_unit_fields(aligned_msgs):

    unit_fields = []

    for col in range(aligned_msgs.shape[1]):
        col_vals = aligned_msgs[:, col]

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
            merged.append({
                "start": uf["start"],
                "end": uf["end"],
                "is_static": False,
            })
            i += 1

    return merged


def build_keyword_candidates(merged_fields):

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

    aligned_np = np.array(alignment, dtype=object)

    unit_fields = detect_unit_fields(aligned_np)
    merged_fields = merge_static_fields(unit_fields)
    keyword_candidates = build_keyword_candidates(merged_fields)

    return unit_fields, merged_fields, keyword_candidates