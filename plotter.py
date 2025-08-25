import argparse, json, os, math, zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

def read_json_stream(path: Path):
    text = path.read_text(encoding="utf-8")
    rows = []
    try:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    except Exception:
        rows = []
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i< n:
        while i<n and text[i].isspace():
            i+=1
        if i>=n:
            break
        obj, idx = dec.raw_decode(text, idx=i)
        rows.append(obj); i = idx
    return rows

def approx_energy_wh_from_power(df: 'pd.DataFrame') -> float:
    if df.empty or "power_watts" not in df or "timestamp" not in df:
        return 0.0
    
    df = df.sort_values("timestamp")
    p = df["power_watts"].astype(float).values
    t = df["timestamp"].astype("int64").values/1e9

    if len(p) < 2:
        return 0.0
    area_ws = 0.0
    for i in range(1, len(p)):
        dt = max(0.0, t[i] - t[i-1])
        area_ws += 0.5 * (p[i] + p[i-1]) * dt #--> W*s
    return area_ws/3600.0 #Wh

def energy_wh_from_mj(df: 'pd.DataFrame') -> float:
    if "energy_mJ" not in df or df["energy_mJ"].dropna().empty:
        return 0.0
    
    series = df["energy_mJ"].dropna().astype(float)
    if series.empty:
        return 0.0
    diff_mJ = series.max() - series.min()
    if diff_mJ < 0:
        return 0.0
    joules = diff_mJ / 1000.0
    return joules/3600.0

def summarize(df: 'pd.DataFrame') -> Dict[str, Any]:
    out = {}
    for col in ["power_watts","gpu_utilization_percent", "memory_used_MB", "cpu_utilization_percent"]:
        if col in df and not df[col].dropna().empty:
            out[f"{col}_avg"] = float(df[col].mean())
            out[f"{col}_max"] = float(df[col].max())
    wh_nvml = energy_wh_from_mj(df)
    wh_int = approx_energy_wh_from_power(df)
    out["energy_Wh_nvml"] = float(wh_nvml)
    out["energy_Wh_integrated"] = float(wh_int)
    out["energy_kJ_integrated"] = float(wh_int*3.6)*1000.0/1000.0 # (Wh * 3600) / 1000
    out["num_samples"] = int(len(df))
    if "elapsed_s" in df.columns:
        out["duration_seconds"] = float(df["elapsed_s"].max() - df["elapsed_s"].min())
    return out

def plot_timeseries(df: 'pd.DataFrame', x_col: str, y: str, title: str, out_path: Path, ylabel: str = None, x_is_elapsed: bool = False):
    if y not in df.columns or df[y].dropna().empty:
        return False
    fig = plt.figure(figsize=(10,4))
    plt.plot(df[x_col], df[y])
    plt.title(title)
    if x_is_elapsed:
        plt.xlabel("Elapsed seconds")
    else:
        plt.xlabel("Time (UTC)")
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True

def make_zip(out_dir: Path, files, zip_name: str = None) -> Path:
    zip_name = zip_name or f"{out_dir.name}.zip"
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            p = Path(fp)
            if p.exists():
                zf.write(p, arcname=p.name)
    return zip_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to telemetry.jsonl")
    ap.add_argument("--gpu-index", type=int, default=None, help="Filter by GPU index (default: use all)")
    ap.add_argument("--out-dir", default=None, help="Directory to write plots and summary (default: alongside src)")
    ap.add_argument("--zip", action="store_true", help="Also create a ZIP with all outputs")
    ap.add_argument("--zip-name", type=str, default=None, help="Optional name for the ZIP (e.g., run_plots.zip)")
    ap.add_argument("--x-axis",dest="x_axis" ,choices=["elapsed","time"], default="elapsed", help="Use 'elapsed' seconds or wall-clock 'time' on x-axis (default: elapsed)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out_dir) if args.out_dir else src.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows = read_json_stream(src)
    if not rows:
        print(f"[Critical ERROR] No rows parsed from {src}")
        return
    df = pd.DataFrame(rows)

    if args.gpu_index is not None and "gpu_index" in df.columns:
        df = df[df["gpu_index"] == args.gpu_index].copy()

    if "timestamp" not in df.columns:
        raise SystemExit("No 'timestamp' in telemetry file.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    if df.empty:
        raise SystemExit("[ERROR] No rows after parsing/filtering; nothing to plot.")
    t0 = df["timestamp"].iloc[0]
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
    

    csv_path = out_dir / "telemetry_timeseries.csv"
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        pass
    

    summary = summarize(df)
    summary_path  = out_dir / "telemetry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    gpu_name = df["gpu_name"].iloc[0] if "gpu_name" in df.columns and not df["gpu_name"].dropna().empty else "GPU"
    gpu_idx = df["gpu_index"].iloc[0] if "gpu_index" in df.columns and not df["gpu_index"].dropna().empty else "NA"
    base = f"{gpu_name} (index {gpu_idx})"

    x_mode = getattr(args, "x_axis", "elapsed")
    x_col = "elapsed_s" if x_mode == "elapsed" else "timestamp"
    x_is_elapsed = (x_mode == "elapsed")
    
    files_out = [str(summary_path), str(csv_path)]

    if plot_timeseries(df, x_col, "power_watts", f"{base} Power (W)", out_dir/"power_watts.png", "Watts", x_is_elapsed):
        files_out.append(str(out_dir/"power_watts.png"))
    if plot_timeseries(df, x_col, "gpu_utilization_percent", f"{base} GPU Utilization (%)", out_dir/"gpu_utilization.png", "Percent", x_is_elapsed):
        files_out.append(str(out_dir/"gpu_utilization.png"))
    if plot_timeseries(df, x_col, "memory_used_MB", f"{base} GPU Memory Used (MB)", out_dir/"gpu_memory_used.png", "MB", x_is_elapsed):
        files_out.append(str(out_dir/"gpu_memory_used.png"))
    if "cpu_utilization_percent" in df.columns and plot_timeseries(df, x_col, "cpu_utilization_percent", f"Host CPU Utilization (%)", out_dir/"cpu_utilization.png", "Percent",x_is_elapsed):
        files_out.append(str(out_dir/"cpu_utilization.png"))

    result = {
        "out_dir": str(out_dir),
        "summary_json": str(summary_path),
        "csv": str(csv_path),
        "x_axis": args.x_axis,
        "plots": [fp for fp in files_out if fp.endswith(".png")]
    }

    if args.zip:
        default_zip = f"{src.stem}-telemetry-plots.zip"
        zip_path = make_zip(out_dir, files_out, zip_name=args.zip_name or default_zip)
        result["zip"] = str(zip_path)

    print(json.dumps(result, indent=2))
    print("[DONE]")

if __name__ == "__main__":
    main()
    


    