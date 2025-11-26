def format_cell(v):
    """Int -> 2-stellige Hex, None -> '--'."""
    if v is None:
        return "--"
    try:
        return f"{int(v):02X}"
    except (ValueError, TypeError):
        return "??"



def show_alignment_block(alignment, used_indices, start_col=0, end_col=40, max_rows=None):
    """
    alignment: List[List[int|None]] – dein finales Alignment
    used_indices: Liste der original Indizes pro Zeile
    start_col, end_col: Spaltenslice, um nicht alles auf einmal zu sehen
    max_rows: optional Anzahl Reihen begrenzen
    """
    n_seqs = len(alignment)
    if max_rows is not None:
        n_seqs = min(n_seqs, max_rows)

    # Sicherheit: end_col nicht über Länge hinaus
    max_len = max(len(row) for row in alignment)
    start_col = max(0, start_col)
    end_col = min(end_col, max_len)

    # Kopfzeile: Spaltennummern
    header = " " * 20  # Platz für "Seq i (orig j): "
    for col in range(start_col, end_col):
        header += f"{col:02d} "
    print(header)

    # Separator
    print("-" * len(header))

    # Zeilen ausgeben
    for i in range(n_seqs):
        row = alignment[i]
        orig_idx = used_indices[i]
        prefix = f"Seq {i:2d} (orig {orig_idx:3d}): "
        line = prefix
        for col in range(start_col, end_col):
            if col < len(row):
                line += format_cell(row[col]) + " "
            else:
                line += "   "
        print(line)


#optional for debugging
def print_alignment(alignment, used_indices):
    print(f"\nAlignment fertig!")
    print(f"   → {len(alignment)} Nachrichten")
    if alignment:
        print(f"   → {len(alignment[0])} Spalten\n")

    # ALLES ausgeben (alle 10 oder 100 oder wie viele du willst)
    for idx, row in enumerate(alignment):
        print(f" Nachricht {idx} (original idx {used_indices[idx]}):")
        print(row)  # komplette Zeile
        print()  # Leerzeile zum Lesen

    show_full_alignment(alignment, used_indices, cols_per_block=32, max_rows=None)
    return
