"""
Aggregate RESULT lines from experiment logs across machines.

Designed to run on the Air controller node, but also works locally.
Pulls logs via rsync, extracts RESULT lines, produces sorted comparison table.

Usage:
    python -m experiments.aggregate                  # rsync + aggregate
    python -m experiments.aggregate --local          # local logs only (no rsync)
    python -m experiments.aggregate --local --dir logs/
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

MACHINES = {
    "narnia": "km5503@narnia.gccis.rit.edu:~/hybridbid/logs/",
    "m4":     "karthikmattu@100.113.39.50:~/hybridbid/logs/",
}

RESULT_PATTERN = re.compile(
    r"RESULT experiment=(?P<experiment>\S+) "
    r"iqm_return=(?P<iqm_return>[\d.\-]+) "
    r"(?:net_return=(?P<net_return>[\d.\-]+) )?"
    r"capture=(?P<capture>[\d.\-]+) "
    r"(?:violations=(?P<violations>\d+) )?"
    r"min=(?P<min>[\d.\-]+) "
    r"max=(?P<max>[\d.\-]+)"
)


def rsync_logs(local_root: Path) -> None:
    for name, src in MACHINES.items():
        dst = local_root / name
        dst.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["rsync", "-avz", "--timeout=10", src, str(dst)],
                check=True, capture_output=True,
            )
            print(f"[rsync] {name} → {dst}")
        except subprocess.CalledProcessError as e:
            print(f"[rsync] {name} FAILED: {e.stderr.decode()[:200]}", file=sys.stderr)


def parse_log(log_path: Path) -> list[dict]:
    results = []
    try:
        with open(log_path) as f:
            for line in f:
                m = RESULT_PATTERN.search(line)
                if m:
                    g = m.groupdict()
                    results.append({
                        "experiment":  g["experiment"],
                        "iqm_return":  float(g["iqm_return"]),
                        "net_return":  float(g["net_return"]) if g["net_return"] else None,
                        "capture":     float(g["capture"]),
                        "violations":  int(g["violations"]) if g["violations"] else None,
                        "min":         float(g["min"]),
                        "max":         float(g["max"]),
                        "log_file":    log_path.name,
                    })
    except OSError:
        pass
    return results


def aggregate(log_dirs: list[Path]) -> pd.DataFrame:
    all_results = []
    for d in log_dirs:
        for log_file in sorted(d.rglob("*.log")):
            all_results.extend(parse_log(log_file))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results).drop_duplicates(subset=["experiment"])
    # Sort by net_return when available (primary metric), iqm_return as fallback for legacy rows.
    sort_key = df["net_return"].fillna(df["iqm_return"])
    df = df.assign(_sort_key=sort_key).sort_values("_sort_key", ascending=False).drop(columns="_sort_key").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--local", action="store_true", help="Skip rsync, use local logs only")
    parser.add_argument("--dir", default="logs", help="Local log directory (default: logs/)")
    parser.add_argument("--remote-root", default="~/hybridbid_logs",
                        help="Root dir for rsynced remote logs")
    args = parser.parse_args()

    local_log_dir = Path(args.dir)
    remote_root = Path(args.remote_root).expanduser()

    log_dirs = [local_log_dir]

    if not args.local:
        rsync_logs(remote_root)
        log_dirs.append(remote_root)

    df = aggregate(log_dirs)

    if df.empty:
        print("No RESULT lines found in any log file.")
        return

    tbx = 870.0
    print(f"\n{'='*86}")
    print(f"{'Experiment':<30} {'Gross $/day':>12} {'Net $/day':>12} {'Viol':>5} {'Capture':>8} {'Min':>8} {'Max':>8}")
    print(f"{'─'*86}")
    for _, row in df.iterrows():
        marker = " ←best" if row.name == 0 else ""
        net_str = f"{row['net_return']:>12.2f}" if pd.notna(row['net_return']) else f"{'—':>12}"
        viol_str = f"{int(row['violations']):>5d}" if pd.notna(row['violations']) else f"{'—':>5}"
        print(
            f"{row['experiment']:<30} "
            f"{row['iqm_return']:>12.2f} "
            f"{net_str} "
            f"{viol_str} "
            f"{row['capture']:>8.4f} "
            f"{row['min']:>8.2f} "
            f"{row['max']:>8.2f}"
            f"{marker}"
        )
    print(f"{'─'*86}")
    print(f"{'TBx baseline':<30} {tbx:>12.2f}")
    print(f"{'='*86}\n")

    out = remote_root / "aggregated_results.csv" if not args.local else local_log_dir / "aggregated_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
