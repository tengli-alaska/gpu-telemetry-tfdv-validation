"""
Prepare embedding vectors + metadata for TensorFlow Embedding Projector.
Upload vectors.tsv and metadata.tsv to https://projector.tensorflow.org

Usage: python embeddings/prepare_embeddings.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUT_DIR = os.path.join(REPO_ROOT, "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

# Load train + eval
train_df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))
eval_df = pd.read_csv(os.path.join(DATA_DIR, "eval_data.csv"))

# Tag time period
train_df["time_period"] = "July (Training)"
eval_df["time_period"] = "August (Eval)"

# Combine
df = pd.concat([train_df, eval_df], ignore_index=True)

# Select numerical features for embedding
feature_cols = [
    "machine_cpu_usr", "machine_cpu_kernel", "machine_cpu",
    "machine_gpu", "gpu_util_per_gpu", "machine_load_1",
    "machine_net_receive", "machine_num_worker"
]

# Drop rows with NaN in feature columns
df_clean = df.dropna(subset=feature_cols).copy()
print(f"Total rows: {len(df)} → Clean: {len(df_clean)}")

# Tag anomalous rows (gpu_util_per_gpu > 100 or < 0)
df_clean["health_status"] = np.where(
    (df_clean["gpu_util_per_gpu"] > 100) | (df_clean["gpu_util_per_gpu"] < 0),
    "Anomalous", "Normal"
)
print(f"Health status: {df_clean['health_status'].value_counts().to_dict()}")

# Normalize features
scaler = MinMaxScaler()
vectors = scaler.fit_transform(df_clean[feature_cols])
print(f"Vector shape: {vectors.shape}")

# Export vectors.tsv (no header, tab-separated)
vectors_path = os.path.join(OUT_DIR, "vectors.tsv")
np.savetxt(vectors_path, vectors, delimiter="\t", fmt="%.6f")
print(f"Saved: {vectors_path}")

# Export metadata.tsv (header + labels)
metadata = df_clean[["gpu_type", "time_period", "health_status"]].copy()
metadata_path = os.path.join(OUT_DIR, "metadata.tsv")
metadata.to_csv(metadata_path, sep="\t", index=False)
print(f"Saved: {metadata_path}")

print(f"""
✅ Embeddings ready for TensorFlow Embedding Projector

Instructions:
  1. Go to https://projector.tensorflow.org
  2. Click "Load" on the left panel
  3. Upload vectors.tsv as "Vectors"
  4. Upload metadata.tsv as "Metadata"  
  5. Explore with t-SNE / PCA:
     - Color by gpu_type → see hardware clustering
     - Color by health_status → anomalies as outliers
     - Color by time_period → visualize temporal drift
  6. key views capture → save to assets/
""")