# GPU Inference Telemetry Validation with TFDV + Embedding Projector

Data validation and high-dimensional visualization pipeline for production GPU cluster telemetry, using **TensorFlow Data Validation (TFDV)** and **TensorFlow Embedding Projector**.

**Dataset**: [Alibaba PAI GPU Cluster Trace v2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) — 6,500+ GPUs across 1,800 machines, published in NSDI'22.

**Lab Reference**: Adapted from [TFDV Lab 3 (C2W1_Assignment)](https://github.com/raminmohammadi/MLOps/blob/main/Labs/Tensorflow_Labs/TFDV_Labs/TFDV_Lab3/C2W1_Assignment.ipynb) — DADS 7305 MLOps, Northeastern University (Prof. Ramin Mohammadi)

---

## Motivation

Production ML systems generate massive streams of GPU telemetry — utilization, power draw, memory usage, job durations. Bad data (sensor glitches, missing readings, out-of-range values) silently degrades model performance and scheduling decisions. As clusters evolve with new hardware and shifting workloads, data drift can invalidate assumptions in existing pipelines.

This project demonstrates automated data validation as the first line of defense in a production ML pipeline, combined with geometric visualization to build intuition about the data's structure.

---

## TFDV Workflow

| Step | What | TFDV API |
|------|------|----------|
| 1 | Load preprocessed GPU telemetry CSVs (train / eval / serving splits) | `pd.read_csv()` |
| 2 | Generate descriptive statistics on training baseline | `tfdv.generate_statistics_from_dataframe()` |
| 3 | Visualize training distributions (numerical + categorical) | `tfdv.visualize_statistics()` |
| 4 | Infer data schema from training statistics | `tfdv.infer_schema()` |
| 5 | Customize schema with hardware-specific constraints: `gpu_util ∈ [0,100]`, `gpu_type` domain, required fields | `tfdv.get_feature()`, `float_domain`, `StringDomain` |
| 6 | Generate eval statistics and overlay with training | `tfdv.visualize_statistics(eval, train)` |
| 7 | Inject synthetic anomalies into eval data (negative util, OOR values, missing fields, unknown GPU type) | `pd.concat()` |
| 8 | Validate eval data against schema — detect all injected + real anomalies | `tfdv.validate_statistics()` |
| 9 | Configure drift comparators and detect temporal drift between trace halves | `drift_comparator.infinity_norm`, `jensen_shannon_divergence` |
| 10 | Validate serving data (resource requests only) — detect schema mismatch | `tfdv.validate_statistics()` |
| 11 | Export embeddings and visualize in TensorFlow Embedding Projector | `projector.tensorflow.org` |

---

## Pipeline Overview

```
Alibaba PAI Trace (7 CSV tables, 16M rows)
    │
    ▼
preprocess.py — join tables, engineer features, temporal split, sample
    │
    ├── training_data.csv   (3,000 rows — July baseline)
    ├── eval_data.csv       (1,500 rows — August drift candidate)
    └── serving_data.csv    (500 rows — task resource requests only)
    │
    ▼
TFDV Notebook — stats, schema, anomaly detection, drift detection
    │
    ▼
Embedding Projector — t-SNE/PCA visualization of telemetry space
```

### Preprocessing Details

The raw Alibaba trace contains 7 CSV tables totaling 16M rows. `preprocess.py` performs:

1. **Join** `pai_machine_metric` (2M rows of time-series telemetry) with `pai_machine_spec` (hardware inventory) on machine ID
2. **Filter** to GPU-equipped machines only (1,814 of 1,897 total)
3. **Engineer features**: `gpu_util_per_gpu` = raw GPU sum / number of GPUs (normalizes to per-GPU average), `duration_sec` = measurement window length
4. **Temporal split**: median timestamp divides the trace into first half (training) and second half (eval) — mirrors how production monitoring validates new data against a historical baseline
5. **Serving split**: built from `pai_task_table` — contains resource *requests* (plan_cpu, plan_mem, plan_gpu) but no utilization measurements, simulating incoming production requests
6. **Sample**: 3,000 training + 1,500 eval + 500 serving rows for interactive TFDV analysis

---

## Key Findings

### Anomaly Detection
- **5 injected anomalies** all detected: negative GPU utilization, out-of-range values, missing required fields, unknown GPU type ("A100"), impossible CPU utilization (>100%)
- **Real data anomaly discovered**: `gpu_util_per_gpu = 207.6%` in the actual Alibaba trace — a genuine measurement artifact caught by our custom schema constraints

### Temporal Drift
- GPU type distribution shifted between trace halves (P100 dropped from 27% → 23%)
- `gpu_util_per_gpu` distribution changed between time periods, detected via L-infinity distance

### Serving Schema Mismatch
- 13 missing columns (utilization metrics absent from resource requests)
- 5 new columns (`plan_cpu`, `plan_mem`, `plan_gpu`, `inst_num`, `status`)
- Correctly simulates the training → serving schema gap in production

### Embedding Projector
- GPU types form distinct clusters in t-SNE space
- Anomalous readings (>100% utilization) appear as geometric outliers
- Temporal drift visible when coloring by time period

---

## Reproducing This Project

### Prerequisites
- **Python 3.10** (TFDV requires ≤3.10)
- **Linux x86_64** (TFDV does not publish macOS/ARM wheels)
- **conda** (recommended for environment management)

### Step 1: Clone and Set Up Environment

```bash
git clone https://github.com/tengli-alaska/gpu-telemetry-tfdv-validation.git
cd gpu-telemetry-tfdv-validation

conda create -n tfdv python=3.10 -y
conda activate tfdv
```

### Step 2: Install Dependencies

```bash
# TensorFlow
pip install tensorflow==2.15.0

# TFDV and its dependencies (pinned versions to avoid resolver issues)
pip install tensorflow-data-validation==1.15.1 --no-deps
pip install tensorflow-metadata==1.15.0 tfx-bsl==1.15.1 --no-deps
pip install joblib "pyfarmhash<0.4" "absl-py<2" "pandas<2" "pyarrow<11,>=10" "protobuf<4.21,>=3.20.3"

# Remaining packages
pip install scikit-learn numpy nbformat jupyter
```

### Step 3: Download Raw Data and Preprocess

```bash
mkdir -p data/raw && cd data/raw

# Download Alibaba PAI GPU Trace v2020 (~1.4 GB total)
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz

# Extract
for file in *.tar.gz; do tar -xzf "$file"; done
cd ../..

# Preprocess: join, split, sample
python data/preprocess.py
```

Output:
- `data/processed/training_data.csv` — 3,000 rows, 14 features
- `data/processed/eval_data.csv` — 1,500 rows, 14 features
- `data/processed/serving_data.csv` — 500 rows, 6 features

### Step 4: Run TFDV Validation

```bash
# Script version (saves HTML visualizations + reports to outputs/)
python notebooks/run_tfdv.py

# Or interactively
jupyter notebook notebooks/tfdv_gpu_telemetry.ipynb
```

### Step 5: Generate Embeddings and Visualize

```bash
python embeddings/prepare_embeddings.py
```

Then:
1. Go to https://projector.tensorflow.org
2. Click **Load** on the left panel
3. Upload `embeddings/vectors.tsv` as Vectors
4. Upload `embeddings/metadata.tsv` as Metadata
5. Select **t-SNE** → Color by `gpu_type`, `health_status`, or `time_period`

---

## Project Structure

```
gpu-telemetry-tfdv-validation/
├── data/
│   ├── raw/                          # Raw Alibaba trace CSVs (.gitignored)
│   ├── processed/
│   │   ├── training_data.csv         # July baseline (3,000 rows)
│   │   ├── eval_data.csv             # August drift (1,500 rows)
│   │   └── serving_data.csv          # Task requests (500 rows)
│   └── preprocess.py                 # Join, feature engineering, split
├── notebooks/
│   ├── tfdv_gpu_telemetry.ipynb      # Main TFDV notebook
│   ├── run_tfdv.py                   # Script version (generates outputs/)
│   └── convert_to_notebook.py        # Auto-generates .ipynb from script
├── embeddings/
│   ├── prepare_embeddings.py         # Export vectors.tsv + metadata.tsv
│   ├── vectors.tsv                   # Embedding vectors for projector
│   └── metadata.tsv                  # Labels: gpu_type, time_period, health
├── schema/
│   └── schema.pbtxt                  # TFDV schema artifact
├── outputs/
│   ├── train_stats.html              # Training statistics visualization
│   ├── train_vs_eval_stats.html      # Train vs eval comparison
│   ├── train_vs_serving_stats.html   # Train vs serving comparison
│   ├── anomalies.txt                 # Anomaly detection report
│   ├── drift_anomalies.txt           # Drift detection report
│   └── serving_anomalies.txt         # Serving validation report
├── assets/                           # Screenshots from Embedding Projector
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Citation

```bibtex
@inproceedings{weng2022mlaas,
  title={MLaaS in the Wild: Workload Analysis and Scheduling in 
         Large-Scale Heterogeneous GPU Clusters},
  author={Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and Wang, Wei 
          and Wang, Cheng and He, Jian and Li, Yong and Zhang, Liping 
          and Lin, Wei and Ding, Yu},
  booktitle={19th USENIX Symposium on Networked Systems Design 
             and Implementation (NSDI'22)},
  year={2022}
}
```

---

## Technologies
- **TFDV** — Schema inference, anomaly detection, drift monitoring
- **TensorFlow Embedding Projector** — t-SNE/PCA visualization
- **Apache Beam** — Distributed statistics computation (via TFDV)
- **pandas / scikit-learn** — Preprocessing and feature engineering