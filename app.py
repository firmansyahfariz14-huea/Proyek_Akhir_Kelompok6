# -*- coding: utf-8 -*-
import streamlit as str
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Analisis Tanaman Padi Sumatera",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style('whitegrid')

# ============================================
# UTILITY FUNCTIONS (Fungsi Pembantu)
# ============================================
def kekuatan_korelasi(r):
    r = abs(r)
    if r >= 0.8:   return "Sangat Kuat"
    elif r >= 0.6: return "Kuat"
    elif r >= 0.4: return "Sedang"
    elif r >= 0.2: return "Lemah"
    else:          return "Sangat Lemah"

@st.cache_data
def load_data(file_path):
    # Menggunakan cache agar data tidak di-load ulang setiap interaksi user
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df

# ============================================
# SIDEBAR & FILE UPLOAD
# ============================================
st.sidebar.title("Navigasi & Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload File CSV Data Padi", type=["csv"])

# Gunakan data default atau minta upload jika belum ada
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info("💡 Silakan unggah file `Data_Tanaman_Padi_Sumatera_version_1.csv` melalui sidebar untuk memulai analisis.")
    st.stop()

# Inisialisasi variabel global setelah data sukses dimuat
kolom_numerik = ['Produksi', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
vars_x = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
hipotesis_dict = {
    "H1": ("Luas Panen",     "Luas panen TIDAK berhubungan dengan produksi padi",     "Luas panen BERHUBUNGAN dengan produksi padi"),
    "H2": ("Curah hujan",    "Curah hujan TIDAK berhubungan dengan produksi padi",    "Curah hujan BERHUBUNGAN dengan produksi padi"),
    "H3": ("Kelembapan",     "Kelembapan TIDAK berhubungan dengan produksi padi",     "Kelembapan BERHUBUNGAN dengan produksi padi"),
    "H4": ("Suhu rata-rata", "Suhu rata-rata TIDAK berhubungan dengan produksi padi", "Suhu rata-rata BERHUBUNGAN dengan produksi padi"),
}

# Menu Pilihan Halaman
menu = st.sidebar.radio("Pilih Halaman Analisis:", [
    "Ikhtisar Data", 
    "Statistika Deskriptif", 
    "Visualisasi Distribusi & Outlier", 
    "Analisis Hubungan & Hipotesis"
])

# ============================================
# HALAMAN 1: IKHTISAR DATA
# ============================================
if menu == "Ikhtisar Data":
    st.title("🌾 Dashboard Analisis Tanaman Padi Sumatera")
    st.subheader("Informasi Dasar Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Baris Data", len(df))
    with col2:
        st.metric("Jumlah Provinsi", df['Provinsi'].nunique())
        
    st.write("**Daftar Provinsi:**", ", ".join(sorted(df['Provinsi'].unique())))
    
    st.write("### Cuplikan Data")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================
# HALAMAN 2: STATISTIKA DESKRIPTIF
# ============================================
elif menu == "Statistika Deskriptif":
    st.title("📊 Statistika Deskriptif")
    
    st.write("### Ukuran Pemusatan & Penyebaran Data")
    hasil = pd.DataFrame({
        'Mean': df[kolom_numerik].mean(),
        'Median': df[kolom_numerik].median(),
        'Modus': [df[c].mode().values[0] for c in kolom_numerik],
        'Range': df[kolom_numerik].max() - df[kolom_numerik].min(),
        'Variance': df[kolom_numerik].var(),
        'Std Dev': df[kolom_numerik].std(),
        'IQR': df[kolom_numerik].quantile(0.75) - df[kolom_numerik].quantile(0.25),
    }).round(2)
    st.dataframe(hasil, use_container_width=True)
    
    st.write("### Ringkasan Statistik Cepat (Describe)")
    st.dataframe(df[kolom_numerik].describe().round(2), use_container_width=True)

# ============================================
# HALAMAN 3: VISUALISASI DISTRIBUSI & OUTLIER
# ============================================
elif menu == "Visualisasi Distribusi & Outlier":
    st.title("📈 Visualisasi Distribusi & Deteksi Outlier")
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Variabel", "Deteksi Outlier (Boxplot)", "Produksi Per Provinsi"])
    
    with tab1:
        st.write("### Distribusi Variabel Numerik")
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for i, col in enumerate(kolom_numerik):
            sns.histplot(df[col], kde=True, ax=axes[i], color='seagreen')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i].axvline(df[col].median(), color='blue', linestyle='--', label='Median')
            axes[i].set_title(f'Distribusi {col}')
            axes[i].legend(fontsize=8)
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab2:
        st.write("### Deteksi Outlier Berdasarkan Boxplot")
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for i, col in enumerate(kolom_numerik):
            sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
            axes[i].set_title(f'Boxplot {col}')
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab3:
        st.write("### Rata-rata Produksi Padi per Provinsi di Sumatera (1993-2020)")
        rata_provinsi = df.groupby('Provinsi')['Produksi'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rata_provinsi.plot(kind='barh', color='seagreen', ax=ax)
        ax.set_xlabel('Rata-rata Produksi (Ton)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**Urutan Rata-rata Produksi (Tertinggi ke Terendah):**")
        st.dataframe(rata_provinsi.sort_values(ascending=False).round(0), use_container_width=True)

# ============================================
# HALAMAN 4: ANALISIS HUBUNGAN & HIPOTESIS
# ============================================
elif menu == "Analisis Hubungan & Hipotesis":
    st.title("🔬 Pengujian Hipotesis & Analisis Korelasi")
    
    # 1. Uji Normalitas
    st.subheader("1. Uji Normalitas (Shapiro-Wilk)")
    normalitas_data = []
    for col in kolom_numerik:
        stat, p = stats.shapiro(df[col])
        hasil_uji = "Normal ✅" if p > 0.05 else "Tidak Normal ❌"
        normalitas_data.append({"Variabel": col, "W Statistic": round(stat, 4), "p-value": f"{p:.6f}", "Hasil": hasil_uji})
    
    st.dataframe(pd.DataFrame(normalitas_data), use_container_width=True)
    st.warning("💡 Kesimpulan: Karena seluruh variabel memiliki p-value < 0.05 (Tidak Normal), maka analisis korelasi dilanjutkan menggunakan metode Non-Parametrik: **Korelasi Spearman**.")

    # Hitung Korelasi Spearman untuk visualisasi selanjutnya
    hasil_korelasi = {}
    ringkasan_data = []
    for (key, (var, h0, h1)) in hipotesis_dict.items():
        rho, p = stats.spearmanr(df[var], df['Produksi'])
        keputusan = "Tolak H0 ✅" if p < 0.05 else "Gagal Tolak H0 ❌"
        kuat = kekuatan_korelasi(rho)
        hasil_korelasi[var] = (rho, p)
        
        keputusan_verbal = h1 if p < 0.05 else h0
        ringkasan_data.append({
            "Hipotesis": key,
            "Variabel": var,
            "Rho (Spearman)": round(rho, 4),
            "P-value": f"{p:.6f}",
            "Kekuatan": kuat,
            "Keputusan H0": keputusan,
            "Kesimpulan": keputusan_verbal
        })

    # 2. Heatmap Korelasi & Scatter Plot
    st.subheader("2. Analisis Visual Hubungan")
    col_plot1, col_plot2 = st.columns([1, 1.2])
    
    with col_plot1:
        st.write("**Heatmap Korelasi (Pearson basis numerik):**")
        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        sns.heatmap(df[kolom_numerik].corr(), annot=True, cmap='YlGnBu', fmt='.2f', ax=ax_heat)
        plt.tight_layout()
        st.pyplot(fig_heat)
        
    with col_plot2:
        st.write("**Scatter Plot Hubungan Variabel vs Produksi:**")
        fig_scat, axes_scat = plt.subplots(2, 2, figsize=(10, 8))
        axes_scat = axes_scat.flatten()
        labels_h = ['H1', 'H2', 'H3', 'H4']
        for i, col in enumerate(vars_x):
            rho, p = hasil_korelasi[col]
            sig_label = "Signifikan" if p < 0.05 else "Tidak Signifikan"
            sns.regplot(x=df[col], y=df['Produksi'], ax=axes_scat[i],
                        scatter_kws={'alpha': 0.5, 'color': 'steelblue'},
                        line_kws={'color': 'red'})
            axes_scat[i].set_title(f'{labels_h[i]}: {col} vs Produksi\nrho = {rho:.3f} | {sig_label}', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_scat)

    # 3. Ringkasan Hipotesis
    st.subheader("3. Ringkasan Hasil Uji Hipotesis Formal")
    st.dataframe(pd.DataFrame(ringkasan_data), use_container_width=True)
