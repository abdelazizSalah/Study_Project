import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

AE_TRAIN_RE = re.compile(
    r"=== Autoencoder Training Results \((?P<ft>[^)]+)\s+feature type\) ===(?P<body>.*?)-{5,}",
    re.DOTALL,
)
AE_EXTRACT_RE = re.compile(
    r"=== Autoencoder Feature Extraction Results \((?P<ft>[^)]+)\s+feature type\) ===(?P<body>.*?)-{5,}",
    re.DOTALL,
)
CLF_RE = re.compile(
    r"=== Classifier Results \((?P<ft>[^)]+)\s+feature type\) ===(?P<body>.*?)-{5,}",
    re.DOTALL,
)


def _find_float(body: str, pattern: str) -> float | None:
    m = re.search(pattern, body)
    return float(m.group(1)) if m else None


def parse_one_report(path: str | Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    rows = []

    # Autoencoder training blocks
    for m in AE_TRAIN_RE.finditer(text):
        ft = m.group("ft").strip().upper()
        body = m.group("body")
        sec = _find_float(body, r"Average runtime\s*:\s*([\d.]+)\s*s")
        ram = _find_float(body, r"Peak RAM.*?:\s*([\d.]+)\s*MB")
        if sec is not None and ram is not None:
            rows.append(dict(part="ae", ft=ft, scenario=None, clf=None, phase="train", sec=sec, ram_mb=ram))

    # Autoencoder extraction blocks
    for m in AE_EXTRACT_RE.finditer(text):
        ft = m.group("ft").strip().upper()
        body = m.group("body")
        sec = _find_float(body, r"Average runtime\s*:\s*([\d.]+)\s*s")
        ram = _find_float(body, r"Peak RAM.*?:\s*([\d.]+)\s*MB")
        if sec is not None and ram is not None:
            rows.append(dict(part="ae", ft=ft, scenario=None, clf=None, phase="extract", sec=sec, ram_mb=ram))

    # Classifier blocks
    for m in CLF_RE.finditer(text):
        ft = m.group("ft").strip().upper()
        body = m.group("body")

        scen_m = re.search(r"Scenario\s*:\s*(\d+)", body)
        clf_m = re.search(r"Classifier\s*:\s*([A-Za-z0-9_+-]+)", body)
        if not scen_m or not clf_m:
            continue

        scenario = int(scen_m.group(1))
        clf = clf_m.group(1).strip().lower()

        tr_sec = _find_float(body, r"Train avg runtime\s*:\s*([\d.]+)\s*s")
        tr_ram = _find_float(body, r"Train peak RAM\s*:\s*([\d.]+)\s*MB")
        te_sec = _find_float(body, r"Test\s*avg runtime\s*:\s*([\d.]+)\s*s")
        te_ram = _find_float(body, r"Test\s*peak RAM\s*:\s*([\d.]+)\s*MB")

        if tr_sec is not None and tr_ram is not None:
            rows.append(dict(part="clf", ft=ft, scenario=scenario, clf=clf, phase="train", sec=tr_sec, ram_mb=tr_ram))
        if te_sec is not None and te_ram is not None:
            rows.append(dict(part="clf", ft=ft, scenario=scenario, clf=clf, phase="test", sec=te_sec, ram_mb=te_ram))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No blocks found in {path}")

    # average duplicates (you had repeated ocsvm/lof blocks in RE15)
    df = (
        df.groupby(["part", "ft", "scenario", "clf", "phase"], dropna=False, as_index=False)
          .agg(sec=("sec", "mean"), ram_mb=("ram_mb", "mean"))
    )
    return df


def plot_for_feature(df: pd.DataFrame, feature_name: str, out_dir: str | Path = "results/plots", log_ram: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-3  # for log scale when value is 0.00

    df = df.copy()
    df["sec_plot"] = df["sec"].clip(lower=eps)
    df["ram_plot"] = df["ram_mb"].clip(lower=eps)

    # -------- Runtime figure --------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ae = df[df["part"] == "ae"].copy()
    if not ae.empty:
        ae = ae.set_index("phase").reindex(["train", "extract"]).reset_index()
        axes[0].bar(ae["phase"], ae["sec_plot"])
        axes[0].set_title(f"Autoencoder Runtime ({feature_name})")
        axes[0].set_ylabel("seconds (log)")
        axes[0].set_yscale("log")
        for i in range(len(ae)):
            axes[0].text(i, ae["sec_plot"].iloc[i], f"{ae['sec'].iloc[i]:.2f}s",
                         ha="center", va="bottom", fontsize=8)
    else:
        axes[0].set_axis_off()
        axes[0].set_title("Autoencoder (no data)")

    clf = df[df["part"] == "clf"].copy()
    if not clf.empty:
        clf["x"] = "s" + clf["scenario"].astype(int).astype(str) + " – " + clf["clf"].astype(str)
        piv = clf.pivot_table(index="x", columns="phase", values="sec_plot", aggfunc="first").fillna(eps)
        piv = piv.reindex(columns=[c for c in ["train", "test"] if c in piv.columns])
        piv.plot(kind="bar", ax=axes[1])
        axes[1].set_title(f"Classifier Runtime ({feature_name})")
        axes[1].set_ylabel("seconds (log)")
        axes[1].set_yscale("log")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].legend(title="phase")
    else:
        axes[1].set_axis_off()
        axes[1].set_title("Classifiers (no data)")

    fig.tight_layout()
    fig.savefig(out_dir / f"runtime_{feature_name.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------- RAM figure --------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if not ae.empty:
        axes[0].bar(ae["phase"], ae["ram_plot"])
        axes[0].set_title(f"Autoencoder Peak RAM ({feature_name})")
        axes[0].set_ylabel("MB" + (" (log)" if log_ram else ""))
        if log_ram:
            axes[0].set_yscale("log")
        for i in range(len(ae)):
            axes[0].text(i, ae["ram_plot"].iloc[i], f"{ae['ram_mb'].iloc[i]/1024:.1f}GB",
                         ha="center", va="bottom", fontsize=8)
    else:
        axes[0].set_axis_off()
        axes[0].set_title("Autoencoder (no data)")

    if not clf.empty:
        clf["x"] = "s" + clf["scenario"].astype(int).astype(str) + " – " + clf["clf"].astype(str)
        piv = clf.pivot_table(index="x", columns="phase", values="ram_plot", aggfunc="first").fillna(eps)
        piv = piv.reindex(columns=[c for c in ["train", "test"] if c in piv.columns])
        piv.plot(kind="bar", ax=axes[1])
        axes[1].set_title(f"Classifier Peak RAM ({feature_name})")
        axes[1].set_ylabel("MB" + (" (log)" if log_ram else ""))
        if log_ram:
            axes[1].set_yscale("log")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].legend(title="phase")
    else:
        axes[1].set_axis_off()
        axes[1].set_title("Classifiers (no data)")

    fig.tight_layout()
    fig.savefig(out_dir / f"ram_{feature_name.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved runtime + ram plots for {feature_name} to {out_dir}")


def plot_all_runtimes():
    raw_path = "results/runtime_raw.txt"
    re15_path = "results/runtime_re15.txt"

    out_dir = "results/plots"

    df_raw = parse_one_report(raw_path)
    plot_for_feature(df_raw, feature_name="RAW", out_dir=out_dir, log_ram=True)

    df_re15 = parse_one_report(re15_path)
    plot_for_feature(df_re15, feature_name="RE15", out_dir=out_dir, log_ram=True)
    return