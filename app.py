import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
pd.set_option('display.width',200)
pd.set_option('display.max_columns',None)

st.set_page_config(page_title="Analisis Produksi Padi Sumatera",layout="wide")
st.title("Analisis Produksi Padi Sumatera")

uploaded_file=st.file_uploader("complete_data.csv",type=["csv"])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file,encoding="utf-8-sig")
    kolom_numerik=['Produksi','Luas Panen','Curah hujan','Kelembapan','Suhu rata-rata']
    st.header("Informasi Dataset")
    c1,c2=st.columns(2)
    c1.metric("Jumlah baris",len(df))
    c2.metric("Jumlah provinsi",df['province'].nunique())
    st.write(sorted(df["province"].unique()))

    st.header("Statistika Deskriptif")
    hasil=pd.DataFrame({
        'Mean': df[kolom_numerik].mean(),
        'Median': df[kolom_numerik].median(),
        'Modus':[df[c].mode().values[0] for c in kolom_numerik],
        'Range':df[kolom_numerik].max()-df[kolom_numerik].min(),
        'Variance':df[kolom_numerik].var(),
        'Std Dev':df[kolom_numerik].std(),
        'IQR':df[kolom_numerik].quantile(.75)-df[kolom_numerik].quantile(.25),
    }).round(2)
    st.dataframe(hasil)
    st.dataframe(df[kolom_numerik].describe().round(2))

    st.header("Visualisasi")
    fig,axes=plt.subplots(2,3,figsize=(16,9));axes=axes.flatten()
    for i,col in enumerate(kolom_numerik):
        sns.histplot(df[col],kde=True,ax=axes[i],color='seagreen')
        axes[i].axvline(df[col].mean(),color='r',ls='--')
        axes[i].axvline(df[col].median(),color='b',ls='--')
        axes[i].set_title(col)
    axes[-1].axis('off');plt.tight_layout();st.pyplot(fig);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(16,9));axes=axes.flatten()
    for i,col in enumerate(kolom_numerik):
        sns.boxplot(y=df[col],ax=axes[i],color='lightcoral');axes[i].set_title(col)
    axes[-1].axis('off');plt.tight_layout();st.pyplot(fig);plt.close(fig)

    rata=df.groupby("Provinsi")["Produksi"].mean().sort_values()
    fig,ax=plt.subplots(figsize=(10,6));rata.plot(kind="barh",ax=ax,color="seagreen")
    st.pyplot(fig);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(df[kolom_numerik].corr(),annot=True,cmap="YlGnBu",fmt=".2f",ax=ax)
    st.pyplot(fig);plt.close(fig)

    st.header("Uji Hipotesis")
    hipotesis={"H1":"Luas Panen","H2":"Curah hujan","H3":"Kelembapan","H4":"Suhu rata-rata"}
    st.subheader("Normalitas Shapiro-Wilk")
    norm=[]
    for col in kolom_numerik:
        W,p=stats.shapiro(df[col]);norm.append([col,W,p,"Normal" if p>0.05 else "Tidak Normal"])
    st.dataframe(pd.DataFrame(norm,columns=["Variabel","W","p-value","Hasil"]))

    def kekuatan(r):
        r=abs(r)
        return "Sangat Kuat" if r>=0.8 else "Kuat" if r>=0.6 else "Sedang" if r>=0.4 else "Lemah" if r>=0.2 else "Sangat Lemah"

    hasil_k={}
    rows=[]
    for h,v in hipotesis.items():
        rho,p=stats.spearmanr(df[v],df["Produksi"])
        hasil_k[v]=(rho,p)
        rows.append([h,v,rho,p,kekuatan(rho),"Tolak H0" if p<0.05 else "Gagal Tolak H0"])
    st.dataframe(pd.DataFrame(rows,columns=["Hipotesis","Variabel","rho","p-value","Kekuatan","Keputusan"]))

    fig,axes=plt.subplots(2,2,figsize=(12,10));axes=axes.flatten()
    for i,col in enumerate(hipotesis.values()):
        rho,p=hasil_k[col]
        sns.regplot(x=df[col],y=df["Produksi"],ax=axes[i],scatter_kws={"alpha":0.5},line_kws={"color":"red"})
        axes[i].set_title(f"{col}\\nrho={rho:.3f}, p={p:.4f}")
    plt.tight_layout();st.pyplot(fig)
