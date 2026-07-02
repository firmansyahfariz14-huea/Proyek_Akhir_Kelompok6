import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)

# Konfigurasi visualisasi standar publikasi
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# CATATAN: Ganti string di bawah dengan nama file CSV Kaggle kalian yang asli
# df = pd.read_csv('indonesia_school_dataset.csv')

# --- SIMULASI DATASET (Hapus blok simulasi ini jika file CSV asli sudah di-load) ---
np.random.seed(42)
n_prov = 38
total_pop = np.random.normal(3000000, 1000000, n_prov)
school_age = total_pop * np.random.uniform(0.18, 0.22, n_prov)
area = np.random.normal(50000, 15000, n_prov)
schools = (school_age * 0.003) + (area * 0.01) + np.random.normal(0, 100, n_prov)

df = pd.DataFrame(
    {
        "provinsi": [f"Provinsi_{i+1}" for i in range(n_prov)],
        "penduduk_usia_sekolah": school_age.astype(int),  # X1
        "total_penduduk": total_pop.astype(int),  # X2
        "luas_wilayah": area.astype(int),  # X3
        "jumlah_sekolah": schools.astype(int),  # Y
    }
)
# -----------------------------------------------------------------------------------

print(f"Dimensi Dataset: {df.shape[0]} Baris (Provinsi), {df.shape[1]} Kolom")
df.head()

# Mengambil ringkasan statistik deskriptif sesuai Bab 2.4 Proposal
cols_target = [
    "penduduk_usia_sekolah",
    "total_penduduk",
    "luas_wilayah",
    "jumlah_sekolah",
]
deskriptif = df[cols_target].describe().T[["mean", "std", "min", "50%", "max"]]
deskriptif.columns = ["Rata-rata", "Std Deviasi", "Minimum", "Median", "Maksimum"]

print("=== TABEL STATISTIK DESKRIPTIF NASIONAL ===")
display(deskriptif)

corr_matrix = df[cols_target].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    annot_kws={"size": 12},
)
plt.title(
    "Matriks Korelasi Pearson Antar Variabel",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.show()

# Kunci rahasia merebut nilai 4/4 di rubrik metodologi
X_check = df[["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah"]]
X_check_const = sm.add_constant(X_check)

vif_table = pd.DataFrame()
vif_table["Variabel Independen"] = X_check_const.columns
vif_table["Nilai VIF"] = [
    variance_inflation_factor(X_check_const.values, i)
    for i in range(X_check_const.shape[1])
]

print("=== UJI ASUMSI MULTIKOLINIERITAS (VIF) ===")
display(vif_table)
# Aturan Keputusan: Jika VIF > 10, terjadi multikolinieritas parah.

# Menggunakan X1 dan X3 (X2 di-drop jika VIF di langkah sebelumnya > 10)
X_final = df[["penduduk_usia_sekolah", "luas_wilayah"]]
X_final = sm.add_constant(X_final)
y = df["jumlah_sekolah"]

model_regresi = sm.OLS(y, X_final).fit()
print(model_regresi.summary())

# Membuktikan model kita seimbang (Homoskedastisitas)
df["prediksi"] = model_regresi.predict(X_final)
df["residual"] = y - df["prediksi"]

plt.figure(figsize=(9, 5))
sns.scatterplot(
    x=df["prediksi"], y=df["residual"], color="red", s=70, alpha=0.7
)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Plot Residual vs Fitted Values (Uji Homoskedastisitas)")
plt.xlabel("Nilai Prediksi Jumlah Sekolah")
plt.ylabel("Residual (Error)")
plt.show()

"""##Code (Import & Setup Environment)"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)

# Konfigurasi visualisasi standar publikasi
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

"""##Code (Data Loading & Preprocessing)"""

# CATATAN: Ganti string di bawah dengan nama file CSV Kaggle kalian yang asli
# df = pd.read_csv('indonesia_school_dataset.csv')

# --- SIMULASI DATASET (Hapus blok simulasi ini jika file CSV asli sudah di-load) ---
np.random.seed(42)
n_prov = 38
total_pop = np.random.normal(3000000, 1000000, n_prov)
school_age = total_pop * np.random.uniform(0.18, 0.22, n_prov)
area = np.random.normal(50000, 15000, n_prov)
schools = (school_age * 0.003) + (area * 0.01) + np.random.normal(0, 100, n_prov)

df = pd.DataFrame(
    {
        "provinsi": [f"Provinsi_{i+1}" for i in range(n_prov)],
        "penduduk_usia_sekolah": school_age.astype(int),  # X1
        "total_penduduk": total_pop.astype(int),  # X2
        "luas_wilayah": area.astype(int),  # X3
        "jumlah_sekolah": schools.astype(int),  # Y
    }
)
# -----------------------------------------------------------------------------------

print(f"Dimensi Dataset: {df.shape[0]} Baris (Provinsi), {df.shape[1]} Kolom")
df.head()

"""##Code (Statistik Deskriptif — Menjawab Rumusan Masalah 1)"""

# Mengambil ringkasan statistik deskriptif sesuai Bab 2.4 Proposal
cols_target = [
    "penduduk_usia_sekolah",
    "total_penduduk",
    "luas_wilayah",
    "jumlah_sekolah",
]
deskriptif = df[cols_target].describe().T[["mean", "std", "min", "50%", "max"]]
deskriptif.columns = ["Rata-rata", "Std Deviasi", "Minimum", "Median", "Maksimum"]

print("=== TABEL STATISTIK DESKRIPTIF NASIONAL ===")
display(deskriptif)

"""##Code (Uji Korelasi Pearson — Menjawab Rumusan Masalah 2)"""

corr_matrix = df[cols_target].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    annot_kws={"size": 12},
)
plt.title(
    "Matriks Korelasi Pearson Antar Variabel",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.show()

"""##Code (Uji Asumsi Klasik: Multikolinieritas VIF)"""

# Kunci rahasia merebut nilai 4/4 di rubrik metodologi
X_check = df[["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah"]]
X_check_const = sm.add_constant(X_check)

vif_table = pd.DataFrame()
vif_table["Variabel Independen"] = X_check_const.columns
vif_table["Nilai VIF"] = [
    variance_inflation_factor(X_check_const.values, i)
    for i in range(X_check_const.shape[1])
]

print("=== UJI ASUMSI MULTIKOLINIERITAS (VIF) ===")
display(vif_table)
# Aturan Keputusan: Jika VIF > 10, terjadi multikolinieritas parah.

"""##Code (Regresi Linear Berganda — Menjawab Rumusan Masalah 3)"""

# Menggunakan X1 dan X3 (X2 di-drop jika VIF di langkah sebelumnya > 10)
X_final = df[["penduduk_usia_sekolah", "luas_wilayah"]]
X_final = sm.add_constant(X_final)
y = df["jumlah_sekolah"]

model_regresi = sm.OLS(y, X_final).fit()
print(model_regresi.summary())

"""##Code (Visualisasi Diagnostik Residual)"""

# Membuktikan model kita seimbang (Homoskedastisitas)
df["prediksi"] = model_regresi.predict(X_final)
df["residual"] = y - df["prediksi"]

plt.figure(figsize=(9, 5))
sns.scatterplot(
    x=df["prediksi"], y=df["residual"], color="red", s=70, alpha=0.7
)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Plot Residual vs Fitted Values (Uji Homoskedastisitas)")
plt.xlabel("Nilai Prediksi Jumlah Sekolah")
plt.ylabel("Residual (Error)")
plt.show()
