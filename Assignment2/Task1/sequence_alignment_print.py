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


def show_full_alignment(alignment, used_indices, cols_per_block=32, max_rows=None):
    """
    Gibt das komplette Alignment blockweise über alle Spalten aus.
    cols_per_block: wie viele Spalten pro Bildschirmblock (z.B. 32 oder 40)
    max_rows: None = alle Sequenzen, sonst begrenzen
    """
    max_len = max(len(row) for row in alignment)
    start = 0
    block_num = 1

    while start < max_len:
        end = min(start + cols_per_block, max_len)
        print(f"\n=== Block {block_num}: Spalten {start}–{end-1} ===")
        show_alignment_block(alignment, used_indices,
                             start_col=start, end_col=end,
                             max_rows=max_rows)
        block_num += 1
        start = end


def show_alignment_block_without_indices(alignment, start_col=0, end_col=40, max_rows=None):
    """
    Hilfsfunktion: Gibt einen Spaltenblock des Alignments aus.
    """

    # 1. Spalten-Header (Indizes)
    header = "Idx |"
    for col in range(start_col, end_col):
        # Formatiert Index als zweistellige Zahl (z.B. 00, 01, 31)
        header += f" {col:02d}"
    print(header)
    print("----+" + "---" * (end_col - start_col))  # Trennlinie

    # 2. Zeilen auswählen (begrenzt durch max_rows)
    rows_to_print = alignment[:max_rows] if max_rows is not None else alignment

    # 3. Nachrichten ausgeben
    for i, row in enumerate(rows_to_print):
        # Zeilenindex (relative Indexnummer im Alignment)
        line = f"{i:3d} |"

        # Werte in der Spalte ausgeben
        for col_idx in range(start_col, end_col):

            # Wichtig: Wir müssen prüfen, ob die Zeile an dieser Spalte
            # lang genug ist, um einen IndexError zu vermeiden.
            if col_idx >= len(row):
                value = None
            else:
                value = row[col_idx]

            if value is None:
                # Lücken (Gaps) als '--'
                line += " --"
            else:
                # Byte-Wert als zweistellige Hex-Zahl (z.B. 0A, FF)
                line += f" {value:02X}"
        print(line)