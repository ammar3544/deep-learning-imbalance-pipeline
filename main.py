import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# Pastikan konsistensi random seed
np.random.seed(42)
tf.random.set_seed(42)

# ================= ================= =================
# 1. GENERATE DATASET UNTUK DEEP LEARNING (Tabular/Sequencing Simulation)
# ================= ================= =================
n_samples = 6000

data = {
    'feature_1': np.random.normal(loc=10.0, scale=2.0, size=n_samples),
    'feature_2': np.random.normal(loc=50.0, scale=10.0, size=n_samples),
    'feature_3': np.random.exponential(scale=5.0, size=n_samples),
    'feature_4': np.random.uniform(low=0.0, high=100.0, size=n_samples),
    # Target: Kasus anomali/minoritas ~10% (Class Imbalance)
    'target': np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10]),
    # FEATURE LEAKAGE (Variabel dari masa depan yang harus dibuang)
    'leakage_feature': np.zeros(n_samples),
}

df = pd.DataFrame(data)
df.loc[df['target'] == 1, 'leakage_feature'] = np.random.uniform(
    500, 1000, size=(df['target'] == 1).sum()
)

print(f'Total Dataset: {len(df)} baris')
print(f"Rasio Kelas Minoritas (Target=1): {df['target'].mean()*100:.2f}%\n")

# ================= ================= =================
# 2. DATA PREPARATION & PENCEGAHAN DATA LEAKAGE
# ================= ================= =================
# WAJIB: Buang kolom 'leakage_feature' dari fitur training
X = df.drop(columns=['target', 'leakage_feature'])
y = df['target']

# Normalisasi fitur (Sangat penting untuk Deep Learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data (Train / Test) dengan Stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================= ================= =================
# 3. MEMBANGUN ARSITEKTUR DEEP LEARNING (MLP)
# ================= ================= =================


def create_model():
  model = models.Sequential([
      layers.Input(shape=(X_train.shape[1],)),
      layers.Dense(32, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(16, activation='relu'),
      layers.Dense(1, activation='sigmoid'),  # Output biner
  ])

  model.compile(
      optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
  )
  return model


print('--- MODEL DEEP LEARNING DENGAN CLASS WEIGHT ---')
# Menghitung class weight secara manual untuk mengatasi imbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

dl_model = create_model()

# Melatih model dengan class weight
history = dl_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=0,  # Ubah ke 1 jika ingin melihat proses per-epoch
)

# Evaluasi Model
y_pred_prob = dl_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Minority/Anomaly (1)']))