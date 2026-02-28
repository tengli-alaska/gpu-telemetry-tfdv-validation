"""
GPU Inference Telemetry Validation with TFDV
Run from repo root: python notebooks/run_tfdv.py

Dataset: Alibaba PAI GPU Cluster Trace v2020 (6,500+ GPUs, NSDI'22)
Adapted from MLOps Lab 3 (C2W1_Assignment) — Ramin Mohammadi, Northeastern
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF CUDA warnings

import tensorflow_data_validation as tfdv
import pandas as pd
import numpy as np
from tensorflow_metadata.proto.v0 import schema_pb2

print(f"TFDV version: {tfdv.__version__}")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
SCHEMA_DIR = os.path.join(REPO_ROOT, "schema")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(SCHEMA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================================
# PHASE 3: Statistics & Schema
# =========================================================================

# ── Step 1: Load datasets ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading datasets")
print("="*60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))
eval_df = pd.read_csv(os.path.join(DATA_DIR, "eval_data.csv"))
serving_df = pd.read_csv(os.path.join(DATA_DIR, "serving_data.csv"))

print(f"Training:  {train_df.shape}")
print(f"Eval:      {eval_df.shape}")
print(f"Serving:   {serving_df.shape}")

print("\n--- Training Data — First 5 Rows ---")
print(train_df.head().to_string())
print("\n--- Column Types ---")
print(train_df.dtypes.to_string())
print("\n--- GPU Type Distribution ---")
print(train_df["gpu_type"].value_counts().to_string())

# ── Step 2: Generate training statistics ─────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Generating training statistics")
print("="*60)

train_stats = tfdv.generate_statistics_from_dataframe(train_df)
print("Training statistics generated")

# Save as HTML for browser viewing
try:
    from tensorflow_data_validation.utils.display_util import get_statistics_html
    html = get_statistics_html(train_stats)
    html_path = os.path.join(OUT_DIR, "train_stats.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"   Saved: {html_path}")
except Exception as e:
    print(f"   (HTML export skipped: {e})")

# Print key stats
print("\n--- Training Statistics Summary ---")
print(train_df.describe().round(2).to_string())

# ── Step 3: Infer schema ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Inferring schema from training data")
print("="*60)

schema = tfdv.infer_schema(train_stats)

print("\n--- Inferred Schema ---")
for feature in schema.feature:
    ftype = "INT" if feature.type == 0 else "FLOAT" if feature.type == 1 else "STRING" if feature.type == 2 else str(feature.type)
    print(f"  {feature.name:25s}  type={ftype:6s}  presence={feature.presence.min_fraction:.2f}")
print("Schema inferred")

# ── Step 4: Customize schema ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Customizing schema for GPU telemetry constraints")
print("="*60)

# 4a. Set gpu_type as categorical with known domain
tfdv.set_domain(schema, "gpu_type", schema_pb2.StringDomain(
    value=["T4", "P100", "V100", "V100M32", "MISC"]
))
print("  gpu_type domain set: T4, P100, V100, V100M32, MISC")

# 4b. Set value ranges for numerical features
gpu_util_feature = tfdv.get_feature(schema, "gpu_util_per_gpu")
gpu_util_feature.float_domain.min = 0.0
gpu_util_feature.float_domain.max = 100.0
print("  gpu_util_per_gpu range: [0, 100]")

cpu_feature = tfdv.get_feature(schema, "machine_cpu")
cpu_feature.float_domain.min = 0.0
cpu_feature.float_domain.max = 100.0
print("  machine_cpu range: [0, 100]")

cpu_usr_feature = tfdv.get_feature(schema, "machine_cpu_usr")
cpu_usr_feature.float_domain.min = 0.0
cpu_usr_feature.float_domain.max = 100.0
print("  machine_cpu_usr range: [0, 100]")

cpu_kernel_feature = tfdv.get_feature(schema, "machine_cpu_kernel")
cpu_kernel_feature.float_domain.min = 0.0
cpu_kernel_feature.float_domain.max = 100.0
print("  machine_cpu_kernel range: [0, 100]")

cpu_iowait_feature = tfdv.get_feature(schema, "machine_cpu_iowait")
cpu_iowait_feature.float_domain.min = 0.0
cpu_iowait_feature.float_domain.max = 100.0
print("  machine_cpu_iowait range: [0, 100]")

cap_gpu_feature = tfdv.get_feature(schema, "cap_gpu")
cap_gpu_feature.int_domain.min = 1
cap_gpu_feature.int_domain.max = 8
print("  cap_gpu range: [1, 8]")

# 4c. Mark required features (presence = 100%)
for feature_name in ["machine_cpu", "gpu_util_per_gpu", "machine_load_1", "gpu_type"]:
    feature = tfdv.get_feature(schema, feature_name)
    feature.presence.min_fraction = 1.0
    print(f"  {feature_name} marked as required (no nulls)")

# Save schema
schema_path = os.path.join(SCHEMA_DIR, "schema.pbtxt")
tfdv.write_schema_text(schema, schema_path)
print(f"\nSchema saved to {schema_path}")

# Print final schema
print("\n--- Customized Schema ---")
for feature in schema.feature:
    ftype = "INT" if feature.type == 0 else "FLOAT" if feature.type == 1 else "STRING" if feature.type == 2 else str(feature.type)
    extras = []
    if feature.HasField("float_domain"):
        extras.append(f"range=[{feature.float_domain.min}, {feature.float_domain.max}]")
    if feature.HasField("int_domain"):
        extras.append(f"range=[{feature.int_domain.min}, {feature.int_domain.max}]")
    if feature.presence.min_fraction == 1.0:
        extras.append("REQUIRED")
    extra_str = "  " + ", ".join(extras) if extras else ""
    print(f"  {feature.name:25s}  type={ftype:6s}{extra_str}")

# =========================================================================
# PHASE 4: Validation & Drift Detection
# =========================================================================

# ── Step 5: Generate eval statistics & compare ───────────────────────────
print("\n" + "="*60)
print("STEP 5: Generating eval statistics & comparing with training")
print("="*60)

eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)
print("Eval statistics generated")

# Save comparison HTML
try:
    html_compare = get_statistics_html(
        eval_stats, train_stats,
        lhs_name="Eval (Second Half)",
        rhs_name="Training (First Half)"
    )
    html_path = os.path.join(OUT_DIR, "train_vs_eval_stats.html")
    with open(html_path, "w") as f:
        f.write(html_compare)
    print(f"   Saved: {html_path}")
except Exception as e:
    print(f"   (HTML export skipped: {e})")

# Print distribution comparison
print("\n--- GPU Type Distribution Comparison ---")
print(f"  {'GPU Type':10s} {'Training':>10s} {'Eval':>10s}")
print(f"  {'-'*10} {'-'*10} {'-'*10}")
train_counts = train_df["gpu_type"].value_counts()
eval_counts = eval_df["gpu_type"].value_counts()
for gpu in sorted(set(train_counts.index) | set(eval_counts.index)):
    t = train_counts.get(gpu, 0)
    e = eval_counts.get(gpu, 0)
    print(f"  {gpu:10s} {t:10d} {e:10d}")

# ── Step 6: Inject synthetic anomalies ───────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Injecting synthetic anomalies into eval data")
print("="*60)

anomalous_rows = pd.DataFrame([
    {
        # Anomaly 1: Negative GPU utilization (sensor glitch)
        "machine_cpu_usr": 15.0, "machine_cpu_kernel": 3.0,
        "machine_cpu_iowait": 0.0, "machine_cpu": 18.0,
        "machine_gpu": -50.0, "gpu_util_per_gpu": -25.0,
        "machine_load_1": 10.0, "machine_net_receive": 1e8,
        "machine_num_worker": 4.0, "gpu_type": "T4",
        "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 2,
        "duration_sec": 300
    },
    {
        # Anomaly 2: GPU utilization > 100% (extreme case)
        "machine_cpu_usr": 30.0, "machine_cpu_kernel": 5.0,
        "machine_cpu_iowait": 0.0, "machine_cpu": 35.0,
        "machine_gpu": 1200.0, "gpu_util_per_gpu": 150.0,
        "machine_load_1": 50.0, "machine_net_receive": 5e8,
        "machine_num_worker": 8.0, "gpu_type": "P100",
        "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 8,
        "duration_sec": 600
    },
    {
        # Anomaly 3: Missing critical values (NaN in required fields)
        "machine_cpu_usr": 20.0, "machine_cpu_kernel": 4.0,
        "machine_cpu_iowait": 0.0, "machine_cpu": np.nan,
        "machine_gpu": 100.0, "gpu_util_per_gpu": np.nan,
        "machine_load_1": np.nan, "machine_net_receive": 2e8,
        "machine_num_worker": 5.0, "gpu_type": "V100",
        "cap_cpu": 64, "cap_mem": 384, "cap_gpu": 2,
        "duration_sec": 450
    },
    {
        # Anomaly 4: Unknown GPU type (new hardware not in training schema)
        "machine_cpu_usr": 25.0, "machine_cpu_kernel": 6.0,
        "machine_cpu_iowait": 0.0, "machine_cpu": 31.0,
        "machine_gpu": 200.0, "gpu_util_per_gpu": 50.0,
        "machine_load_1": 20.0, "machine_net_receive": 3e8,
        "machine_num_worker": 6.0, "gpu_type": "A100",
        "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 4,
        "duration_sec": 500
    },
    {
        # Anomaly 5: CPU utilization > 100% (impossible value)
        "machine_cpu_usr": 105.0, "machine_cpu_kernel": 8.0,
        "machine_cpu_iowait": 0.0, "machine_cpu": 113.0,
        "machine_gpu": 80.0, "gpu_util_per_gpu": 40.0,
        "machine_load_1": 30.0, "machine_net_receive": 2e8,
        "machine_num_worker": 4.0, "gpu_type": "T4",
        "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 2,
        "duration_sec": 350
    },
])

eval_df_with_anomalies = pd.concat([eval_df, anomalous_rows], ignore_index=True)
print(f"Eval data: {len(eval_df)} → {len(eval_df_with_anomalies)} rows (+{len(anomalous_rows)} anomalies)")
print("\nInjected anomalies:")
print("  1. Negative GPU utilization (gpu_util_per_gpu = -25)")
print("  2. GPU utilization > 100% (gpu_util_per_gpu = 150)")
print("  3. Missing required values (machine_cpu, gpu_util_per_gpu, machine_load_1 = NaN)")
print("  4. Unknown GPU type ('A100' not in training schema domain)")
print("  5. CPU utilization > 100% (machine_cpu = 113)")

# ── Step 7: Validate eval with anomalies ─────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Validating eval data (with anomalies) against schema")
print("="*60)

eval_stats_anomalous = tfdv.generate_statistics_from_dataframe(eval_df_with_anomalies)
anomalies = tfdv.validate_statistics(eval_stats_anomalous, schema)

print("\n--- Anomalies Detected ---")
if anomalies.anomaly_info:
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        print(f"\n   Feature: {feature_name}")
        print(f"     Type: {anomaly_info.short_description}")
        print(f"     Detail: {anomaly_info.description}")
else:
    print("  No anomalies detected (unexpected!)")

# Save anomalies report
anomalies_path = os.path.join(OUT_DIR, "anomalies.txt")
with open(anomalies_path, "w") as f:
    f.write("TFDV Anomaly Detection Report\n")
    f.write("="*50 + "\n")
    f.write(f"Dataset: Eval data with {len(anomalous_rows)} injected anomalies\n\n")
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        f.write(f"Feature: {feature_name}\n")
        f.write(f"  Type: {anomaly_info.short_description}\n")
        f.write(f"  Detail: {anomaly_info.description}\n\n")
print(f"\nSaved: {anomalies_path}")

# ── Step 8: Drift detection ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Detecting drift between training and eval (clean)")
print("="*60)

# Set drift comparators — numeric features use L-infinity
for feat_name in ["gpu_util_per_gpu", "machine_cpu", "machine_load_1"]:
    feat = tfdv.get_feature(schema, feat_name)
    feat.drift_comparator.infinity_norm.threshold = 0.01
    print(f"  ✓ {feat_name}: L-infinity threshold = 0.01")

# Categorical drift uses Jensen-Shannon divergence
gpu_type_feat = tfdv.get_feature(schema, "gpu_type")
gpu_type_feat.drift_comparator.jensen_shannon_divergence.threshold = 0.01
print("  ✓ gpu_type: Jensen-Shannon divergence threshold = 0.01")

drift_anomalies = tfdv.validate_statistics(
    eval_stats, schema, previous_statistics=train_stats
)

print("\n--- Drift Detected ---")
if drift_anomalies.anomaly_info:
    for feature_name, anomaly_info in drift_anomalies.anomaly_info.items():
        print(f"\n   Feature: {feature_name}")
        print(f"     Type: {anomaly_info.short_description}")
        print(f"     Detail: {anomaly_info.description}")
else:
    print("  No drift detected")

# Save drift report
drift_path = os.path.join(OUT_DIR, "drift_anomalies.txt")
with open(drift_path, "w") as f:
    f.write("TFDV Drift Detection Report\n")
    f.write("="*50 + "\n")
    f.write("Comparison: Training (first half) vs Eval (second half)\n\n")
    if drift_anomalies.anomaly_info:
        for feature_name, anomaly_info in drift_anomalies.anomaly_info.items():
            f.write(f"Feature: {feature_name}\n")
            f.write(f"  Type: {anomaly_info.short_description}\n")
            f.write(f"  Detail: {anomaly_info.description}\n\n")
    else:
        f.write("No drift detected.\n")
print(f"Saved: {drift_path}")

# ── Step 9: Validate serving data ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: Validating serving data against training schema")
print("="*60)

print(f"\nServing data shape: {serving_df.shape}")
print(f"Serving columns: {list(serving_df.columns)}")
print(f"Training columns: {list(train_df.columns)}")
print(f"\nMissing in serving: {set(train_df.columns) - set(serving_df.columns)}")

serving_stats = tfdv.generate_statistics_from_dataframe(serving_df)

# Save comparison HTML
try:
    html_serving = get_statistics_html(
        serving_stats, train_stats,
        lhs_name="Serving (Resource Requests)",
        rhs_name="Training (Utilization Metrics)"
    )
    html_path = os.path.join(OUT_DIR, "train_vs_serving_stats.html")
    with open(html_path, "w") as f:
        f.write(html_serving)
    print(f"   Saved: {html_path}")
except Exception as e:
    print(f"   (HTML export skipped: {e})")

serving_anomalies = tfdv.validate_statistics(serving_stats, schema)

print("\n--- Serving Data Anomalies ---")
if serving_anomalies.anomaly_info:
    for feature_name, anomaly_info in serving_anomalies.anomaly_info.items():
        print(f"\n   Feature: {feature_name}")
        print(f"     Type: {anomaly_info.short_description}")
        print(f"     Detail: {anomaly_info.description}")
else:
    print("  No anomalies detected")

# Save serving report
serving_path = os.path.join(OUT_DIR, "serving_anomalies.txt")
with open(serving_path, "w") as f:
    f.write("TFDV Serving Data Validation Report\n")
    f.write("="*50 + "\n")
    f.write(f"Serving columns: {list(serving_df.columns)}\n")
    f.write(f"Training columns: {list(train_df.columns)}\n")
    f.write(f"Missing in serving: {set(train_df.columns) - set(serving_df.columns)}\n\n")
    if serving_anomalies.anomaly_info:
        for feature_name, anomaly_info in serving_anomalies.anomaly_info.items():
            f.write(f"Feature: {feature_name}\n")
            f.write(f"  Type: {anomaly_info.short_description}\n")
            f.write(f"  Detail: {anomaly_info.description}\n\n")
    else:
        f.write("No anomalies detected.\n")
print(f"Saved: {serving_path}")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPLETE — All outputs saved")
print("="*60)
print(f"""
Outputs:
  Schema:      {schema_path}
  Stats HTML:  {OUT_DIR}/train_stats.html
               {OUT_DIR}/train_vs_eval_stats.html
               {OUT_DIR}/train_vs_serving_stats.html
  Reports:     {OUT_DIR}/anomalies.txt
               {OUT_DIR}/drift_anomalies.txt
               {OUT_DIR}/serving_anomalies.txt

Open the .html files in a browser to see interactive TFDV visualizations.
""")