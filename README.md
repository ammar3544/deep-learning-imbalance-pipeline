# Deep Learning Pipeline: Handling Imbalance & Data Leakage

Repositori ini berisi studi kasus penerapan model *Deep Learning* dengan fokus pada pencegahan *data leakage* dan penanganan *class imbalance* (berdasarkan materi Analisis Data & AI - BRIN).

## Fitur Proyek
- **Eliminasi Data Leakage**: Membuang fitur pasca-kejadian (`leakage_feature`).
- **Feature Scaling**: Standardisasi data menggunakan `StandardScaler` agar optimal untuk jaringan syaraf tiruan (*Neural Network*).
- **Class Weight Adjustment**: Menyesuaikan bobot kelas minoritas pada model *Deep Learning* agar metrik *Recall* optimal.

## Cara Menjalankan
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan skrip: `python main.py`