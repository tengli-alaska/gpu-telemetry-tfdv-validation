"""
Preprocess Alibaba PAI GPU Cluster Trace v2020 for TFDV validation lab.

Produces 3 CSVs:
  - training_data.csv   (first half of trace — baseline)
  - eval_data.csv       (second half of trace — drift candidate)
  - serving_data.csv    (task resource requests — no utilization labels)

Usage:
    python preprocess.py
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column headers (from .header files — CSVs have no headers)
# ---------------------------------------------------------------------------
MACHINE_METRIC_COLS = [
    "worker_name", "machine", "start_time", "end_time",
    "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
    "machine_gpu", "machine_load_1", "machine_net_receive",
    "machine_num_worker", "machine_cpu"
]

MACHINE_SPEC_COLS = [
    "machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"
]

SENSOR_COLS = [
    "job_name", "task_name", "worker_name", "inst_id", "machine",
    "gpu_name", "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
    "avg_gpu_wrk_mem", "max_gpu_wrk_mem", "read", "write",
    "read_count", "write_count"
]

TASK_COLS = [
    "job_name", "task_name", "inst_num", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem",
    "plan_gpu", "gpu_type"
]

# ---------------------------------------------------------------------------
# 1. Load machine specs (small — 1,897 rows)
# ---------------------------------------------------------------------------
print("Loading machine specs...")
specs = pd.read_csv(
    os.path.join(RAW_DIR, "pai_machine_spec.csv"),
    header=None, names=MACHINE_SPEC_COLS
)
# Keep only GPU machines (gpu_type != "CPU" and cap_gpu > 0)
gpu_specs = specs[specs["gpu_type"] != "CPU"].copy()
print(f"  Total machines: {len(specs)}, GPU machines: {len(gpu_specs)}")
print(f"  GPU types: {gpu_specs['gpu_type'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 2. Load machine metrics (2M rows — sample for speed)
# ---------------------------------------------------------------------------
print("\nLoading machine metrics...")
metrics = pd.read_csv(
    os.path.join(RAW_DIR, "pai_machine_metric.csv"),
    header=None, names=MACHINE_METRIC_COLS,
    dtype={"machine": str, "worker_name": str}
)
print(f"  Raw rows: {len(metrics):,}")

# Drop rows with all-null metric values
metric_cols = [
    "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
    "machine_gpu", "machine_load_1", "machine_net_receive",
    "machine_num_worker", "machine_cpu"
]
metrics[metric_cols] = metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
metrics.dropna(subset=metric_cols, how="all", inplace=True)
print(f"  After dropping all-null rows: {len(metrics):,}")

# Convert timestamps to numeric
metrics["start_time"] = pd.to_numeric(metrics["start_time"], errors="coerce")
metrics["end_time"] = pd.to_numeric(metrics["end_time"], errors="coerce")

# ---------------------------------------------------------------------------
# 3. Join metrics with machine specs
# ---------------------------------------------------------------------------
print("\nJoining metrics with machine specs...")
merged = metrics.merge(gpu_specs, on="machine", how="inner")
print(f"  Rows after join (GPU machines only): {len(merged):,}")

# Normalize GPU utilization: machine_gpu / cap_gpu * 100
# machine_gpu is summed across GPUs, cap_gpu is number of GPUs
merged["gpu_util_per_gpu"] = np.where(
    merged["cap_gpu"] > 0,
    merged["machine_gpu"] / merged["cap_gpu"],
    0.0
)

# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
print("\nEngineering features...")
merged["duration_sec"] = merged["end_time"] - merged["start_time"]

# Select final features for TFDV
feature_cols = [
    "machine_cpu_usr",       # CPU user utilization %
    "machine_cpu_kernel",    # CPU kernel utilization %
    "machine_cpu_iowait",    # CPU IO wait %
    "machine_cpu",           # Overall CPU utilization %
    "machine_gpu",           # Raw GPU utilization (summed across GPUs)
    "gpu_util_per_gpu",      # Normalized GPU utilization per GPU
    "machine_load_1",        # 1-min load average
    "machine_net_receive",   # Network receive bytes
    "machine_num_worker",    # Number of workers on machine
    "gpu_type",              # Categorical: GPU type
    "cap_cpu",               # Machine CPU capacity
    "cap_mem",               # Machine memory capacity
    "cap_gpu",               # Number of GPUs
    "start_time",            # For temporal splitting
    "duration_sec",          # Measurement window duration
]
df = merged[feature_cols].copy()

# Drop any remaining NaN rows in critical columns
critical = ["machine_cpu", "gpu_util_per_gpu", "machine_load_1"]
df.dropna(subset=critical, inplace=True)
print(f"  Clean rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 5. Temporal split
# ---------------------------------------------------------------------------
print("\nSplitting by time period...")
mid_time = df["start_time"].median()
print(f"  Time range: {df['start_time'].min():.0f} — {df['start_time'].max():.0f}")
print(f"  Midpoint: {mid_time:.0f}")

train_df = df[df["start_time"] <= mid_time].copy()
eval_df = df[df["start_time"] > mid_time].copy()

print(f"  Training rows (first half): {len(train_df):,}")
print(f"  Eval rows (second half): {len(eval_df):,}")

# ---------------------------------------------------------------------------
# 6. Sample to manageable size
# ---------------------------------------------------------------------------
TRAIN_SAMPLE = 3000
EVAL_SAMPLE = 1500

print(f"\nSampling: train={TRAIN_SAMPLE}, eval={EVAL_SAMPLE}")
if len(train_df) > TRAIN_SAMPLE:
    train_df = train_df.sample(n=TRAIN_SAMPLE, random_state=42)
if len(eval_df) > EVAL_SAMPLE:
    eval_df = eval_df.sample(n=EVAL_SAMPLE, random_state=42)

# Drop start_time from final output (used only for splitting)
train_df = train_df.drop(columns=["start_time"])
eval_df = eval_df.drop(columns=["start_time"])

# ---------------------------------------------------------------------------
# 7. Build serving split from task table
# ---------------------------------------------------------------------------
print("\nBuilding serving split from task table...")
tasks = pd.read_csv(
    os.path.join(RAW_DIR, "pai_task_table.csv"),
    header=None, names=TASK_COLS,
    dtype={"job_name": str, "task_name": str}
)

# Convert numeric columns
for c in ["inst_num", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu"]:
    tasks[c] = pd.to_numeric(tasks[c], errors="coerce")

# Keep only GPU tasks (plan_gpu > 0)
gpu_tasks = tasks[tasks["plan_gpu"] > 0].copy()
print(f"  Total tasks: {len(tasks):,}, GPU tasks: {len(gpu_tasks):,}")

# Select serving features (resource REQUESTS only — no utilization labels)
serving_cols = ["plan_cpu", "plan_mem", "plan_gpu", "gpu_type", "inst_num", "status"]
serving_df = gpu_tasks[serving_cols].copy()

# Sample serving data
SERVING_SAMPLE = 500
if len(serving_df) > SERVING_SAMPLE:
    serving_df = serving_df.sample(n=SERVING_SAMPLE, random_state=42)

print(f"  Serving rows: {len(serving_df):,}")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
train_path = os.path.join(OUT_DIR, "training_data.csv")
eval_path = os.path.join(OUT_DIR, "eval_data.csv")
serving_path = os.path.join(OUT_DIR, "serving_data.csv")

train_df.to_csv(train_path, index=False)
eval_df.to_csv(eval_path, index=False)
serving_df.to_csv(serving_path, index=False)

print(f"\n✅ Saved:")
print(f"  {train_path} ({len(train_df):,} rows, {len(train_df.columns)} cols)")
print(f"  {eval_path} ({len(eval_df):,} rows, {len(eval_df.columns)} cols)")
print(f"  {serving_path} ({len(serving_df):,} rows, {len(serving_df.columns)} cols)")

# ---------------------------------------------------------------------------
# 9. Quick summary stats
# ---------------------------------------------------------------------------
print("\n--- Training Data Summary ---")
print(train_df.describe().round(2).to_string())
print(f"\ngpu_type distribution:\n{train_df['gpu_type'].value_counts().to_string()}")

print("\n--- Eval Data Summary ---")
print(f"gpu_type distribution:\n{eval_df['gpu_type'].value_counts().to_string()}")

print("\n--- Serving Data Summary ---")
print(f"gpu_type distribution:\n{serving_df['gpu_type'].value_counts().to_string()}")
print(f"status distribution:\n{serving_df['status'].value_counts().to_string()}")