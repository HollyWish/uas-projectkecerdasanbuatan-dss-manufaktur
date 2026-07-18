import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


st.set_page_config(
    page_title="DSS Quality Control",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(" Sistem Cerdas Pengendalian Kualitas (Quality Control)")
st.markdown("Pendekatan *Unsupervised Machine Learning* untuk Segmentasi Cacat Produk Manufaktur")
st.markdown("---")

#Data Set
@st.cache_data
def load_data():
    df = pd.read_csv("defects_data.csv")
    severity_mapping = {"Minor": 1, "Moderate": 2, "Critical": 3}
    if "severity_score" not in df.columns:
        df["severity_score"] = df["severity"].map(severity_mapping)
    return df

df = load_data()

#Tab
tab_profil, tab_data, tab_kmeans, tab_hc, tab_keputusan = st.tabs([
    "Identitas Proyek",
    "Eksplorasi Data",
    "Pemodelan K-Means",
    "Pemodelan Hierarki",
    "Tindak Lanjut Manajerial"
])


with tab_profil:
    st.header("Latar Belakang Proyek")
    st.write("""
    Aplikasi Sistem Pendukung Keputusan (DSS) ini dikembangkan untuk mengotomatisasi proses pengelompokan data riwayat cacat produk di lantai produksi.
    Dengan mengimplementasikan algoritma clustering, sistem ini diharapkan mampu mengidentifikasi pola anomali mesin yang sulit diamati secara manual.
     Hasil analisis dari dashboard ini dapat dimanfaatkan sebagai dasar pertimbangan dalam mengambil langkah Predictive Maintenance yang lebih tepat sasaran..
    """)

    st.info("""
    **Identitas Pengembang:**
    * **Nama:** Rizal Kurniawan
    * **NIM:** E1202401954
    * **Program Studi:** Teknik Industri
    """)


with tab_data:
    st.header("Tinjauan Dataset & Analisis Deskriptif")

    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Total Observasi", f"{df.shape[0]} Baris")
    col_met2.metric("Total Variabel", f"{df.shape[1]} Kolom")
    col_met3.metric("Rata-rata Biaya Perbaikan", f"${df['repair_cost'].mean():.2f}")

    st.write("**Pratinjau Data Mentah:**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("##Visualisasi Distribusi Utama")
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='severity', palette='mako', ax=ax1, order=['Minor', 'Moderate', 'Critical'])
        ax1.set_title("Proporsi Tingkat Keparahan")
        ax1.set_xlabel("Tingkat Keparahan (Severity)")
        ax1.set_ylabel("Frekuensi")
        st.pyplot(fig1)

    with col_plot2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='defect_type', palette='rocket', ax=ax2)
        ax2.set_title("Proporsi Jenis Cacat")
        ax2.set_xlabel("Jenis Cacat (Defect Type)")
        ax2.set_ylabel("Frekuensi")
        st.pyplot(fig2)


with tab_kmeans:
    st.header("Algoritma K-Means & Simulasi Prediksi")

    Preprocessing
    df_kmeans = df.dropna(subset=["repair_cost"]).copy()
    X_km = df_kmeans[["repair_cost", "severity_score"]]
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(X_km)

    Modeling
    model_km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_kmeans["Cluster"] = model_km.fit_predict(X_scaled_km)

    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        fig_km, ax_km = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_kmeans, x="repair_cost", y="severity_score", hue="Cluster", palette="Set1", s=100, edgecolor='white', ax=ax_km)
        ax_km.set_title("Pemetaan Klaster K-Means")
        ax_km.set_xlabel("Biaya Perbaikan ($)")
        ax_km.set_ylabel("Skor Keparahan")
        ax_km.set_yticks([1, 2, 3])
        ax_km.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
        st.pyplot(fig_km)

    with col_k2:
        st.write("**Karakteristik Klaster (Rata-rata):**")
        st.dataframe(df_kmeans.groupby("Cluster")[["repair_cost", "severity_score"]].mean())

    st.markdown("##Uji Coba Model (Data Baru)")
    with st.form("form_kmeans"):
        c1, c2 = st.columns(2)
        in_cost = c1.number_input("Masukkan Estimasi Biaya Perbaikan ($)", min_value=0.0, max_value=2000.0, value=300.0)
        in_sev = c2.selectbox("Pilih Estimasi Keparahan", ["Minor", "Moderate", "Critical"])

        if st.form_submit_button("Jalankan Klasifikasi"):
            mapping = {"Minor": 1, "Moderate": 2, "Critical": 3}
            new_data = scaler_km.transform([[in_cost, mapping[in_sev]]])
            hasil = model_km.predict(new_data)[0]

            if in_sev == "Critical":
                st.error(f"⚠️ **TERDETEKSI ANOMALI KRITIS (Masuk Klaster {hasil}). Hentikan lini produksi untuk inspeksi!**")
            else:
                st.success(f"✅ **KONDISI AMAN (Masuk Klaster {hasil}). Proses produksi dapat dilanjutkan.**")


with tab_hc:
    st.header("Algoritma Hierarchical (Agglomerative) Clustering")

    Preprocessing
    df_hc = df.dropna(subset=["repair_cost"]).copy()
    X_hc = df_hc[["repair_cost", "severity_score"]]
    scaler_hc = StandardScaler()
    X_scaled_hc = scaler_hc.fit_transform(X_hc)

    Modeling
    model_hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    df_hc["Cluster_Hierarki"] = model_hc.fit_predict(X_scaled_hc)

    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        fig_hc, ax_hc = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_hc, x="repair_cost", y="severity_score", hue="Cluster_Hierarki", palette="Dark2", s=100, edgecolor='white', ax=ax_hc)
        ax_hc.set_title("Pemetaan Klaster Hierarki")
        ax_hc.set_xlabel("Biaya Perbaikan ($)")
        ax_hc.set_ylabel("Skor Keparahan")
        ax_hc.set_yticks([1, 2, 3])
        ax_hc.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
        st.pyplot(fig_hc)

    with col_h2:
        st.write("**Karakteristik Klaster (Rata-rata):**")
        st.dataframe(df_hc.groupby("Cluster_Hierarki")[["repair_cost", "severity_score"]].mean())


with tab_keputusan:
    st.header("Penarikan Kesimpulan & Tindak Lanjut Manajerial")

    st.markdown("##Diagnosis Karakteristik Klaster")
    st.write("""
    Berdasarkan output pemodelan analitik, teridentifikasi tiga kelompok utama cacat produksi:
    *   **Klaster Toleransi (Minor/Biaya Rendah):** Insiden rutin yang tidak membahayakan sistem operasional.
    *   **Klaster Peringatan (Moderat):** Indikasi awal penurunan performa mesin, membutuhkan kalibrasi ulang alat.
    *   **Klaster Bahaya (Kritis):** Kegagalan struktural yang memakan biaya besar dan menyebabkan *downtime* signifikan.
    """)

    st.markdown("##Usulan Kebijakan Strategis")
    st.info("""
    1. **Transisi ke Predictive Maintenance:** Tinggalkan sistem perbaikan reaktif. Jadwalkan perawatan mesin secara berkala berdasarkan prediksi anomali dari algoritma klastering di atas.
    2. **Investigasi Root-Cause:** Lakukan evaluasi khusus pada lini permesinan yang secara historis paling sering menghasilkan produk pada *Klaster Bahaya*.
    3. **Optimalisasi Quality Control (QC):** Perketat metode inspeksi pada komponen yang menunjukkan gejala cacat moderat sebelum berevolusi menjadi cacat kritis.
    """)
