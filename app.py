# ============================================================
# APLIKASI KLASIFIKASI TINGKAT KEDISIPLINAN KARYAWAN
# Integrasi Data Fingerprint dan Buku Kegiatan Mandor (BKM)
# Menggunakan Algoritma Naive Bayes
#
# PT PP London Sumatra Indonesia Tbk - Pahu Makmur Estate
#
# VERSI:
# - 2 file upload dinamis
# - Tidak menentukan bulan secara manual
# - Periode dibaca otomatis dari BKM Date
# - File bebas: Januari-Februari, Juli-Agustus, Oktober-November, dll.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Kedisiplinan Karyawan",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CSS TAMBAHAN
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f6ff;
        border: 1px solid #d5e5ff;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. JUDUL APLIKASI
# ============================================================

st.markdown(
    '<div class="main-title">🌴 Klasifikasi Tingkat Kedisiplinan Karyawan</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Integrasi Data Fingerprint dan Buku Kegiatan Mandor (BKM)
    menggunakan Algoritma Naive Bayes
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Menu Aplikasi")

    menu = st.radio(
        "Pilih Menu",
        [
            "🏠 Dashboard",
            "📊 Data Karyawan",
            "👤 Detail Karyawan",
            "🤖 Evaluasi Naive Bayes",
            "ℹ️ Tentang Aplikasi"
        ]
    )

    st.divider()

    st.markdown("### 📋 Keterangan")

    st.write(
        """
        Aplikasi ini digunakan untuk mengintegrasikan data
        Fingerprint dan Buku Kegiatan Mandor (BKM) untuk
        melakukan klasifikasi tingkat kedisiplinan karyawan.
        """
    )

    st.divider()

    st.caption(
        "Data diproses secara lokal melalui aplikasi."
    )


# ============================================================
# 6. FUNGSI MEMBERSIHKAN NAMA KOLOM
# ============================================================

def bersihkan_nama_kolom(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# 7. FUNGSI MENCARI KOLOM
# ============================================================

def cari_kolom(df, kandidat):

    kolom_df = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for nama in kandidat:

        nama_lower = nama.strip().lower()

        if nama_lower in kolom_df:
            return kolom_df[nama_lower]

    return None


# ============================================================
# 8. FUNGSI MEMBACA EXCEL
# ============================================================

def baca_excel(file):

    try:

        df = pd.read_excel(file)

        df = bersihkan_nama_kolom(df)

        return df

    except Exception as e:

        st.error(
            f"❌ Gagal membaca file {file.name}: {e}"
        )

        return None


# ============================================================
# 9. FUNGSI MENGUBAH WAKTU MENJADI DETIK
# ============================================================

def waktu_ke_detik(nilai):

    if pd.isna(nilai):
        return np.nan

    # Jika berupa datetime/time
    if hasattr(nilai, "hour"):

        try:

            return (
                nilai.hour * 3600
                + nilai.minute * 60
                + nilai.second
            )

        except:
            pass

    # Jika berupa angka Excel
    if isinstance(nilai, (int, float)):

        try:

            if 0 <= float(nilai) < 1:

                return float(nilai) * 24 * 3600

        except:
            pass

    # Jika string
    try:

        teks = str(nilai).strip()

        # Format HH:MM
        match = re.match(
            r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$",
            teks
        )

        if match:

            jam = int(match.group(1))
            menit = int(match.group(2))

            detik = (
                int(match.group(3))
                if match.group(3)
                else 0
            )

            return (
                jam * 3600
                + menit * 60
                + detik
            )

    except:
        pass

    return np.nan


# ============================================================
# 10. FUNGSI DETEKSI PERIODE DATA
# ============================================================

def deteksi_periode(df):

    kolom_tanggal = cari_kolom(
        df,
        [
            "BKM Date",
            "Date",
            "Tanggal",
            "BKM_Date"
        ]
    )

    if kolom_tanggal is None:

        return "Periode tidak diketahui"

    tanggal = pd.to_datetime(
        df[kolom_tanggal],
        errors="coerce"
    )

    tanggal_valid = tanggal.dropna()

    if len(tanggal_valid) == 0:

        return "Periode tidak diketahui"

    tanggal_min = tanggal_valid.min()
    tanggal_max = tanggal_valid.max()

    nama_bulan = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember"
    }

    # Jika semua tanggal berada di bulan yang sama
    if (
        tanggal_min.month == tanggal_max.month
        and tanggal_min.year == tanggal_max.year
    ):

        return (
            f"{nama_bulan[tanggal_min.month]} "
            f"{tanggal_min.year}"
        )

    # Jika mencakup lebih dari satu bulan
    return (
        f"{nama_bulan[tanggal_min.month]} "
        f"{tanggal_min.year} - "
        f"{nama_bulan[tanggal_max.month]} "
        f"{tanggal_max.year}"
    )


# ============================================================
# 11. FUNGSI MENAMBAHKAN PERIODE
# ============================================================

def tambahkan_periode(df):

    df = df.copy()

    kolom_tanggal = cari_kolom(
        df,
        [
            "BKM Date",
            "Date",
            "Tanggal",
            "BKM_Date"
        ]
    )

    if kolom_tanggal is None:

        df["Periode"] = "Tidak diketahui"

        return df

    df["Tanggal_Proses"] = pd.to_datetime(
        df[kolom_tanggal],
        errors="coerce"
    )

    df["Periode"] = (
        df["Tanggal_Proses"]
        .dt.strftime("%B %Y")
    )

    # Mengubah nama bulan Inggris menjadi Indonesia
    bulan_indonesia = {
        "January": "Januari",
        "February": "Februari",
        "March": "Maret",
        "April": "April",
        "May": "Mei",
        "June": "Juni",
        "July": "Juli",
        "August": "Agustus",
        "September": "September",
        "October": "Oktober",
        "November": "November",
        "December": "Desember"
    }

    for inggris, indonesia in bulan_indonesia.items():

        df["Periode"] = (
            df["Periode"]
            .str.replace(
                inggris,
                indonesia,
                regex=False
            )
        )

    return df


# ============================================================
# 12. FUNGSI MENYIAPKAN DATA
# ============================================================

def siapkan_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Employee
    # --------------------------------------------------------

    kolom_employee = cari_kolom(
        df,
        [
            "Employee",
            "Employee ID",
            "Employee_ID",
            "NIK",
            "NIP"
        ]
    )

    if kolom_employee is None:

        raise ValueError(
            "Kolom Employee tidak ditemukan."
        )

    if kolom_employee != "Employee":

        df["Employee"] = df[kolom_employee]

    df["Employee"] = (
        df["Employee"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Employee Name
    # --------------------------------------------------------

    kolom_nama = cari_kolom(
        df,
        [
            "Employee Name",
            "Employee_Name",
            "Nama Karyawan",
            "Nama"
        ]
    )

    if kolom_nama is None:

        df["Employee Name"] = "Tidak Diketahui"

    elif kolom_nama != "Employee Name":

        df["Employee Name"] = df[kolom_nama]

    df["Employee Name"] = (
        df["Employee Name"]
        .fillna("Tidak Diketahui")
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # BKM Date
    # --------------------------------------------------------

    kolom_tanggal = cari_kolom(
        df,
        [
            "BKM Date",
            "Date",
            "Tanggal",
            "BKM_Date"
        ]
    )

    if kolom_tanggal is None:

        raise ValueError(
            "Kolom BKM Date / Tanggal tidak ditemukan."
        )

    if kolom_tanggal != "BKM Date":

        df["BKM Date"] = df[kolom_tanggal]

    df["BKM Date"] = pd.to_datetime(
        df["BKM Date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Time In
    # --------------------------------------------------------

    kolom_masuk = cari_kolom(
        df,
        [
            "Time In",
            "TimeIn",
            "Check In",
            "Check_In",
            "Jam Masuk"
        ]
    )

    if kolom_masuk is None:

        df["Time In"] = np.nan

    elif kolom_masuk != "Time In":

        df["Time In"] = df[kolom_masuk]


    # --------------------------------------------------------
    # Time Out
    # --------------------------------------------------------

    kolom_keluar = cari_kolom(
        df,
        [
            "Time Out",
            "TimeOut",
            "Check Out",
            "Check_Out",
            "Jam Keluar"
        ]
    )

    if kolom_keluar is None:

        df["Time Out"] = np.nan

    elif kolom_keluar != "Time Out":

        df["Time Out"] = df[kolom_keluar]


    # --------------------------------------------------------
    # BKM Number
    # --------------------------------------------------------

    kolom_bkm = cari_kolom(
        df,
        [
            "BKM Number",
            "BKM_Number",
            "Nomor BKM",
            "No BKM"
        ]
    )

    if kolom_bkm is None:

        df["BKM Number"] = np.nan

    elif kolom_bkm != "BKM Number":

        df["BKM Number"] = df[kolom_bkm]


    # --------------------------------------------------------
    # Act Code
    # --------------------------------------------------------

    kolom_act = cari_kolom(
        df,
        [
            "Act Code",
            "Act_Code",
            "Activity Code",
            "Kode Aktivitas"
        ]
    )

    if kolom_act is None:

        df["Act Code"] = "Tidak Diketahui"

    elif kolom_act != "Act Code":

        df["Act Code"] = df[kolom_act]

    df["Act Code"] = (
        df["Act Code"]
        .fillna("Tidak Diketahui")
        .astype(str)
    )


    # --------------------------------------------------------
    # Kolom numerik
    # --------------------------------------------------------

    daftar_numerik = {

        "Manday": [
            "Manday",
            "Man Day",
            "Man_Day"
        ],

        "Output": [
            "Output"
        ],

        "Premi": [
            "Premi",
            "Premium"
        ],

        "Penalty": [
            "Penalty",
            "Pinalti"
        ],

        "Overtime": [
            "Overtime",
            "Over Time"
        ],

        "1-Valid 0-Non": [
            "1-Valid 0-Non",
            "Valid",
            "Valid BKM",
            "1-Valid"
        ]
    }


    for nama_baru, kandidat in daftar_numerik.items():

        kolom = cari_kolom(
            df,
            kandidat
        )

        if kolom is None:

            df[nama_baru] = 0

        elif kolom != nama_baru:

            df[nama_baru] = df[kolom]

        df[nama_baru] = pd.to_numeric(
            df[nama_baru],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # Waktu masuk
    # --------------------------------------------------------

    df["TimeIn_Detik"] = (
        df["Time In"]
        .apply(waktu_ke_detik)
    )

    df["TimeOut_Detik"] = (
        df["Time Out"]
        .apply(waktu_ke_detik)
    )


    # --------------------------------------------------------
    # Durasi kerja
    # --------------------------------------------------------

    df["Durasi_Jam"] = (
        df["TimeOut_Detik"]
        - df["TimeIn_Detik"]
    ) / 3600


    # Jika durasi negatif, kemungkinan melewati tengah malam
    df.loc[
        df["Durasi_Jam"] < 0,
        "Durasi_Jam"
    ] += 24


    # Durasi yang tidak masuk akal dianggap kosong
    df.loc[
        df["Durasi_Jam"] > 24,
        "Durasi_Jam"
    ] = np.nan


    # --------------------------------------------------------
    # Fingerprint hadir
    # --------------------------------------------------------

    df["Fingerprint_Hadir"] = (
        df["Time In"]
        .notna()
        .astype(int)
    )


    # --------------------------------------------------------
    # Periode otomatis
    # --------------------------------------------------------

    df = tambahkan_periode(df)


    return df


# ============================================================
# 13. FUNGSI MEMPROSES DATA
# ============================================================

def proses_data(data_1, data_2):

    # --------------------------------------------------------
    # Siapkan data
    # --------------------------------------------------------

    data_1 = siapkan_data(data_1)
    data_2 = siapkan_data(data_2)


    # --------------------------------------------------------
    # Beri identitas sumber file
    # --------------------------------------------------------

    data_1["Sumber_File"] = "File 1"
    data_2["Sumber_File"] = "File 2"


    # --------------------------------------------------------
    # Gabungkan
    # --------------------------------------------------------

    data = pd.concat(
        [
            data_1,
            data_2
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Pastikan tanggal valid
    # --------------------------------------------------------

    data = data[
        data["BKM Date"].notna()
    ].copy()


    if len(data) == 0:

        raise ValueError(
            "Tidak terdapat tanggal yang valid pada kolom BKM Date."
        )


    # ========================================================
    # DATA HARIAN
    # ========================================================

    data_harian = (
        data
        .groupby(
            [
                "Employee",
                "Employee Name",
                "BKM Date"
            ],
            as_index=False
        )
        .agg(

            Fingerprint_Hadir=(
                "Fingerprint_Hadir",
                "max"
            ),

            Durasi_Jam=(
                "Durasi_Jam",
                "mean"
            ),

            Total_BKM_Harian=(
                "BKM Number",
                "nunique"
            ),

            Total_Manday_Harian=(
                "Manday",
                "sum"
            ),

            Total_Output_Harian=(
                "Output",
                "sum"
            ),

            Total_Premi_Harian=(
                "Premi",
                "sum"
            ),

            Total_Penalty_Harian=(
                "Penalty",
                "sum"
            ),

            Total_Overtime_Harian=(
                "Overtime",
                "sum"
            ),

            Total_Valid_BKM_Harian=(
                "1-Valid 0-Non",
                "sum"
            )
        )
    )


    # ========================================================
    # TAMBAHKAN PERIODE KE DATA HARIAN
    # ========================================================

    data_harian["Periode"] = (
        data_harian["BKM Date"]
        .dt.to_period("M")
        .astype(str)
    )


    # ========================================================
    # DATA BULANAN PER KARYAWAN
    # ========================================================

    data_bulanan = (
        data_harian
        .groupby(
            [
                "Employee",
                "Employee Name",
                "Periode"
            ],
            as_index=False
        )
        .agg(

            Hari_Tercatat=(
                "BKM Date",
                "nunique"
            ),

            Hari_Hadir=(
                "Fingerprint_Hadir",
                "sum"
            ),

            Rata_Durasi_Jam=(
                "Durasi_Jam",
                "mean"
            ),

            Total_BKM=(
                "Total_BKM_Harian",
                "sum"
            ),

            Total_Manday=(
                "Total_Manday_Harian",
                "sum"
            ),

            Total_Output=(
                "Total_Output_Harian",
                "sum"
            ),

            Total_Premi=(
                "Total_Premi_Harian",
                "sum"
            ),

            Total_Penalty=(
                "Total_Penalty_Harian",
                "sum"
            ),

            Total_Overtime=(
                "Total_Overtime_Harian",
                "sum"
            ),

            Total_Valid_BKM=(
                "Total_Valid_BKM_Harian",
                "sum"
            )
        )
    )


    # ========================================================
    # RASIO KEHADIRAN
    # ========================================================

    data_bulanan["Rasio_Kehadiran"] = np.where(

        data_bulanan["Hari_Tercatat"] > 0,

        data_bulanan["Hari_Hadir"]
        / data_bulanan["Hari_Tercatat"],

        0
    )


    # ========================================================
    # KATEGORI KEDISIPLINAN
    # ========================================================

    def kategori_disiplin(rasio):

        if rasio >= 0.90:

            return "Tinggi"

        elif rasio >= 0.75:

            return "Sedang"

        else:

            return "Rendah"


    data_bulanan["Kategori_Kedisiplinan"] = (
        data_bulanan["Rasio_Kehadiran"]
        .apply(kategori_disiplin)
    )


    # ========================================================
    # FITUR FREKUENSI ACT CODE
    # ========================================================

    aktivitas = (
        data
        .groupby(
            [
                "Employee",
                "Employee Name",
                "BKM Date",
                "Act Code"
            ]
        )
        .size()
        .reset_index(
            name="Frekuensi"
        )
    )


    aktivitas["Periode"] = (
        aktivitas["BKM Date"]
        .dt.to_period("M")
        .astype(str)
    )


    aktivitas_bulanan = (
        aktivitas
        .groupby(
            [
                "Employee",
                "Employee Name",
                "Periode",
                "Act Code"
            ]
        )["Frekuensi"]
        .sum()
        .reset_index()
    )


    if len(aktivitas_bulanan) > 0:

        aktivitas_pivot = (
            aktivitas_bulanan
            .pivot_table(
                index=[
                    "Employee",
                    "Employee Name",
                    "Periode"
                ],
                columns="Act Code",
                values="Frekuensi",
                fill_value=0
            )
            .reset_index()
        )


        # Beri nama kolom aktivitas
        kolom_baru = []

        for col in aktivitas_pivot.columns:

            if col in [
                "Employee",
                "Employee Name",
                "Periode"
            ]:

                kolom_baru.append(col)

            else:

                nama_aman = (
                    "Act_"
                    + re.sub(
                        r"[^A-Za-z0-9_]+",
                        "_",
                        str(col)
                    )
                )

                kolom_baru.append(
                    nama_aman
                )

        aktivitas_pivot.columns = kolom_baru


        # Gabungkan
        data_bulanan = data_bulanan.merge(
            aktivitas_pivot,
            on=[
                "Employee",
                "Employee Name",
                "Periode"
            ],
            how="left"
        )


    # ========================================================
    # ISI NILAI KOSONG
    # ========================================================

    data_bulanan = data_bulanan.fillna(0)


    # ========================================================
    # TAMPILKAN PERIODE DALAM FORMAT INDONESIA
    # ========================================================

    def format_periode(periode):

        try:

            tahun, bulan = periode.split("-")

            nama_bulan = {
                "01": "Januari",
                "02": "Februari",
                "03": "Maret",
                "04": "April",
                "05": "Mei",
                "06": "Juni",
                "07": "Juli",
                "08": "Agustus",
                "09": "September",
                "10": "Oktober",
                "11": "November",
                "12": "Desember"
            }

            return (
                f"{nama_bulan.get(bulan, bulan)} "
                f"{tahun}"
            )

        except:

            return str(periode)


    data_bulanan["Periode_Tampilan"] = (
        data_bulanan["Periode"]
        .apply(format_periode)
    )


    # ========================================================
    # FITUR MODEL
    # ========================================================

    fitur_dasar = [

        "Rata_Durasi_Jam",
        "Total_BKM",
        "Total_Manday",
        "Total_Output",
        "Total_Premi",
        "Total_Penalty",
        "Total_Overtime",
        "Total_Valid_BKM"

    ]


    # Tambahkan fitur Act
    fitur_aktivitas = [
        col
        for col in data_bulanan.columns
        if str(col).startswith("Act_")
    ]


    fitur = (
        fitur_dasar
        + fitur_aktivitas
    )


    # Pastikan fitur tersedia
    fitur = [
        col
        for col in fitur
        if col in data_bulanan.columns
    ]


    if len(fitur) == 0:

        raise ValueError(
            "Tidak terdapat fitur yang dapat digunakan untuk model."
        )


    # ========================================================
    # X DAN Y
    # ========================================================

    X = data_bulanan[fitur].copy()

    y = data_bulanan[
        "Kategori_Kedisiplinan"
    ].copy()


    # Ubah semua fitur menjadi numerik
    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


    # ========================================================
    # CEK JUMLAH KATEGORI
    # ========================================================

    jumlah_kelas = y.nunique()


    if jumlah_kelas < 2:

        return (
            data_bulanan,
            None,
            None,
            data,
            fitur
        )


    # ========================================================
    # PEMBAGIAN DATA TRAINING DAN TESTING
    # ========================================================

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )


    # ========================================================
    # PIPELINE NAIVE BAYES
    # ========================================================

    model = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "naive_bayes",
                GaussianNB()
            )
        ]
    )


    # ========================================================
    # TRAINING MODEL
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDIKSI DATA TESTING
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # EVALUASI
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "Tinggi",
            "Sedang",
            "Rendah"
        ]
    )


    laporan = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )


    evaluasi = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "Confusion Matrix": cm,

        "Classification Report": laporan,

        "Jumlah Data Training": len(X_train),

        "Jumlah Data Testing": len(X_test)

    }


    # ========================================================
    # PREDIKSI SELURUH DATA
    # ========================================================

    data_bulanan["Prediksi_Naive_Bayes"] = (
        model.predict(X)
    )


    # ========================================================
    # PROBABILITAS
    # ========================================================

    try:

        probabilitas = (
            model.predict_proba(X)
        )

        kelas_model = (
            model.named_steps[
                "naive_bayes"
            ].classes_
        )


        for i, kelas in enumerate(
            kelas_model
        ):

            data_bulanan[
                f"Probabilitas_{kelas}"
            ] = probabilitas[:, i]


    except:

        pass


    return (
        data_bulanan,
        model,
        evaluasi,
        data,
        fitur
    )


# ============================================================
# 14. BAGIAN UPLOAD DATA
# ============================================================

st.markdown(
    '<div class="section-title">📂 Upload Data</div>',
    unsafe_allow_html=True
)

st.info(
    """
    Silakan upload 2 file data. 
    File dapat berasal dari bulan apa saja.
    Sistem akan membaca periode secara otomatis berdasarkan
    tanggal pada kolom BKM Date.
    """
)


col1, col2 = st.columns(2)


with col1:

    st.markdown("### 📄 File Data 1")

    file_1 = st.file_uploader(
        "Upload file pertama",
        type=[
            "xlsx",
            "xls"
        ],
        key="file_data_1"
    )


with col2:

    st.markdown("### 📄 File Data 2")

    file_2 = st.file_uploader(
        "Upload file kedua",
        type=[
            "xlsx",
            "xls"
        ],
        key="file_data_2"
    )


# ============================================================
# 15. CEK FILE
# ============================================================

if file_1 is not None and file_2 is not None:

    st.success(
        "✅ Kedua file berhasil diupload."
    )


    # --------------------------------------------------------
    # Baca file untuk mendeteksi periode
    # --------------------------------------------------------

    try:

        preview_1 = pd.read_excel(
            file_1
        )

        preview_2 = pd.read_excel(
            file_2
        )


        periode_1 = deteksi_periode(
            preview_1
        )

        periode_2 = deteksi_periode(
            preview_2
        )


        st.markdown(
            "### 📅 Periode Data Terdeteksi"
        )


        pcol1, pcol2 = st.columns(2)


        with pcol1:

            st.info(
                f"📄 File 1\n\n"
                f"**{periode_1}**"
            )


        with pcol2:

            st.info(
                f"📄 File 2\n\n"
                f"**{periode_2}**"
            )


    except Exception as e:

        st.warning(
            f"Periode belum dapat dibaca: {e}"
        )


    # ========================================================
    # TOMBOL PROSES
    # ========================================================

    if st.button(
        "🚀 PROSES DATA",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Sedang memproses data dan menjalankan Naive Bayes..."
        ):

            try:

                # --------------------------------------------
                # Baca Excel
                # --------------------------------------------

                data_1 = pd.read_excel(
                    file_1
                )

                data_2 = pd.read_excel(
                    file_2
                )


                # --------------------------------------------
                # Proses
                # --------------------------------------------

                (
                    dataset,
                    model,
                    evaluasi,
                    data_gabungan,
                    fitur
                ) = proses_data(
                    data_1,
                    data_2
                )


                # --------------------------------------------
                # Simpan ke session
                # --------------------------------------------

                st.session_state[
                    "dataset"
                ] = dataset

                st.session_state[
                    "model"
                ] = model

                st.session_state[
                    "evaluasi"
                ] = evaluasi

                st.session_state[
                    "data_gabungan"
                ] = data_gabungan

                st.session_state[
                    "fitur"
                ] = fitur

                st.session_state[
                    "periode_file_1"
                ] = periode_1

                st.session_state[
                    "periode_file_2"
                ] = periode_2


                st.success(
                    "✅ Data berhasil diproses!"
                )


            except Exception as e:

                st.error(
                    f"❌ Terjadi kesalahan saat memproses data:\n\n"
                    f"{type(e).__name__}: {e}"
                )

                st.exception(e)


# ============================================================
# 16. AMBIL DATA DARI SESSION
# ============================================================

dataset = st.session_state.get(
    "dataset",
    None
)

model = st.session_state.get(
    "model",
    None
)

evaluasi = st.session_state.get(
    "evaluasi",
    None
)

data_gabungan = st.session_state.get(
    "data_gabungan",
    None
)

fitur = st.session_state.get(
    "fitur",
    []
)


# ============================================================
# 17. DASHBOARD
# ============================================================

if dataset is not None:

    if menu == "🏠 Dashboard":

        st.markdown(
            '<div class="section-title">📊 Dashboard</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Informasi periode
        # ----------------------------------------------------

        st.markdown(
            "### 📅 Periode Data"
        )


        periode_unik = (
            dataset[
                "Periode_Tampilan"
            ]
            .drop_duplicates()
            .tolist()
        )


        st.write(
            ", ".join(
                periode_unik
            )
        )


        # ----------------------------------------------------
        # Metric
        # ----------------------------------------------------

        total_karyawan = (
            dataset[
                "Employee"
            ]
            .nunique()
        )


        total_data = len(
            dataset
        )


        rata_kehadiran = (
            dataset[
                "Rasio_Kehadiran"
            ]
            .mean()
        )


        jumlah_tinggi = (
            dataset[
                "Kategori_Kedisiplinan"
            ]
            .eq("Tinggi")
            .sum()
        )


        jumlah_sedang = (
            dataset[
                "Kategori_Kedisiplinan"
            ]
            .eq("Sedang")
            .sum()
        )


        jumlah_rendah = (
            dataset[
                "Kategori_Kedisiplinan"
            ]
            .eq("Rendah")
            .sum()
        )


        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.metric(
                "👥 Total Karyawan",
                total_karyawan
            )


        with m2:

            st.metric(
                "📋 Data Karyawan",
                total_data
            )


        with m3:

            st.metric(
                "📈 Rata-rata Kehadiran",
                f"{rata_kehadiran * 100:.2f}%"
            )


        with m4:

            if evaluasi is not None:

                st.metric(
                    "🎯 Akurasi Model",
                    f"{evaluasi['Accuracy'] * 100:.2f}%"
                )

            else:

                st.metric(
                    "🎯 Akurasi Model",
                    "N/A"
                )


        st.divider()


        # ----------------------------------------------------
        # Kategori
        # ----------------------------------------------------

        st.markdown(
            "### 📌 Kategori Kedisiplinan"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "🟢 Tinggi",
                jumlah_tinggi
            )


        with c2:

            st.metric(
                "🟡 Sedang",
                jumlah_sedang
            )


        with c3:

            st.metric(
                "🔴 Rendah",
                jumlah_rendah
            )


        # ----------------------------------------------------
        # Grafik
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Distribusi Kedisiplinan"
        )


        distribusi = (
            dataset[
                "Kategori_Kedisiplinan"
            ]
            .value_counts()
            .reindex(
                [
                    "Tinggi",
                    "Sedang",
                    "Rendah"
                ],
                fill_value=0
            )
        )


        st.bar_chart(
            distribusi
        )


        # ----------------------------------------------------
        # Catatan metodologi
        # ----------------------------------------------------

        st.info(
            """
            **Kriteria klasifikasi kedisiplinan:**

            🟢 **Tinggi** = rasio kehadiran ≥ 90%

            🟡 **Sedang** = rasio kehadiran 75% sampai < 90%

            🔴 **Rendah** = rasio kehadiran < 75%

            Kategori tersebut merupakan **label penelitian**
            yang dibentuk berdasarkan rasio kehadiran dan bukan
            klaim sebagai kategori resmi perusahaan.
            """
        )


# ============================================================
# 18. DATA KARYAWAN
# ============================================================

if dataset is not None:

    if menu == "📊 Data Karyawan":

        st.markdown(
            '<div class="section-title">📊 Data Karyawan</div>',
            unsafe_allow_html=True
        )


        st.write(
            "Berikut hasil pengolahan data berdasarkan periode yang terdeteksi otomatis."
        )


        # ----------------------------------------------------
        # Filter periode
        # ----------------------------------------------------

        daftar_periode = [
            "Semua"
        ] + sorted(
            dataset[
                "Periode_Tampilan"
            ]
            .drop_duplicates()
            .tolist()
        )


        periode_pilihan = st.selectbox(
            "📅 Pilih Periode",
            daftar_periode
        )


        data_tampil = dataset.copy()


        if periode_pilihan != "Semua":

            data_tampil = data_tampil[
                data_tampil[
                    "Periode_Tampilan"
                ]
                == periode_pilihan
            ]


        # ----------------------------------------------------
        # Tabel
        # ----------------------------------------------------

        kolom_tampil = [

            "Employee",
            "Employee Name",
            "Periode_Tampilan",
            "Hari_Tercatat",
            "Hari_Hadir",
            "Rasio_Kehadiran",
            "Rata_Durasi_Jam",
            "Total_BKM",
            "Total_Manday",
            "Total_Output",
            "Total_Premi",
            "Total_Penalty",
            "Total_Overtime",
            "Kategori_Kedisiplinan",
            "Prediksi_Naive_Bayes"

        ]


        kolom_tampil = [
            col
            for col in kolom_tampil
            if col in data_tampil.columns
        ]


        tabel = data_tampil[
            kolom_tampil
        ].copy()


        if "Rasio_Kehadiran" in tabel.columns:

            tabel[
                "Rasio_Kehadiran"
            ] = (
                tabel[
                    "Rasio_Kehadiran"
                ] * 100
            ).round(2)


        st.dataframe(
            tabel,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 19. DETAIL KARYAWAN
# ============================================================

if dataset is not None:

    if menu == "👤 Detail Karyawan":

        st.markdown(
            '<div class="section-title">👤 Detail Karyawan</div>',
            unsafe_allow_html=True
        )


        daftar_karyawan = (
            dataset[
                [
                    "Employee",
                    "Employee Name"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "Employee"
            )
        )


        pilihan_karyawan = st.selectbox(
            "🔎 Pilih Karyawan",
            daftar_karyawan[
                "Employee"
            ].tolist()
        )


        detail = dataset[
            dataset[
                "Employee"
            ].astype(str)
            == str(pilihan_karyawan)
        ].copy()


        if len(detail) > 0:

            nama = detail[
                "Employee Name"
            ].iloc[0]


            st.markdown(
                f"### 👤 {nama}"
            )


            st.write(
                f"**Employee ID:** {pilihan_karyawan}"
            )


            # ------------------------------------------------
            # Statistik
            # ------------------------------------------------

            rata = (
                detail[
                    "Rasio_Kehadiran"
                ]
                .mean()
            )


            total_bkm = (
                detail[
                    "Total_BKM"
                ]
                .sum()
            )


            total_output = (
                detail[
                    "Total_Output"
                ]
                .sum()
            )


            d1, d2, d3 = st.columns(3)


            with d1:

                st.metric(
                    "📈 Rata-rata Kehadiran",
                    f"{rata * 100:.2f}%"
                )


            with d2:

                st.metric(
                    "📋 Total BKM",
                    f"{total_bkm:,.0f}"
                )


            with d3:

                st.metric(
                    "📦 Total Output",
                    f"{total_output:,.2f}"
                )


            st.divider()


            # ------------------------------------------------
            # Detail
            # ------------------------------------------------

            kolom_detail = [

                "Employee",
                "Employee Name",
                "Periode_Tampilan",
                "Hari_Tercatat",
                "Hari_Hadir",
                "Rasio_Kehadiran",
                "Rata_Durasi_Jam",
                "Total_BKM",
                "Total_Manday",
                "Total_Output",
                "Total_Premi",
                "Total_Penalty",
                "Total_Overtime",
                "Kategori_Kedisiplinan",
                "Prediksi_Naive_Bayes"

            ]


            kolom_detail = [
                col
                for col in kolom_detail
                if col in detail.columns
            ]


            tabel_detail = detail[
                kolom_detail
            ].copy()


            if "Rasio_Kehadiran" in tabel_detail.columns:

                tabel_detail[
                    "Rasio_Kehadiran"
                ] = (
                    tabel_detail[
                        "Rasio_Kehadiran"
                    ] * 100
                ).round(2)


            st.dataframe(
                tabel_detail,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 20. EVALUASI NAIVE BAYES
# ============================================================

if dataset is not None:

    if menu == "🤖 Evaluasi Naive Bayes":

        st.markdown(
            '<div class="section-title">🤖 Evaluasi Model Naive Bayes</div>',
            unsafe_allow_html=True
        )


        if model is None or evaluasi is None:

            st.warning(
                """
                Model Naive Bayes belum dapat dibuat.

                Kemungkinan seluruh data hanya menghasilkan
                satu kategori kedisiplinan.
                """
            )


        else:

            # ------------------------------------------------
            # Nilai evaluasi
            # ------------------------------------------------

            e1, e2, e3, e4 = st.columns(4)


            with e1:

                st.metric(
                    "Accuracy",
                    f"{evaluasi['Accuracy'] * 100:.2f}%"
                )


            with e2:

                st.metric(
                    "Precision",
                    f"{evaluasi['Precision'] * 100:.2f}%"
                )


            with e3:

                st.metric(
                    "Recall",
                    f"{evaluasi['Recall'] * 100:.2f}%"
                )


            with e4:

                st.metric(
                    "F1-Score",
                    f"{evaluasi['F1-Score'] * 100:.2f}%"
                )


            st.divider()


            # ------------------------------------------------
            # Train/Test
            # ------------------------------------------------

            st.markdown(
                "### 📚 Pembagian Data"
            )


            t1, t2 = st.columns(2)


            with t1:

                st.metric(
                    "Data Training",
                    evaluasi[
                        "Jumlah Data Training"
                    ]
                )


            with t2:

                st.metric(
                    "Data Testing",
                    evaluasi[
                        "Jumlah Data Testing"
                    ]
                )


            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            st.markdown(
                "### 🔲 Confusion Matrix"
            )


            cm = evaluasi[
                "Confusion Matrix"
            ]


            cm_df = pd.DataFrame(

                cm,

                index=[
                    "Aktual Tinggi",
                    "Aktual Sedang",
                    "Aktual Rendah"
                ],

                columns=[
                    "Prediksi Tinggi",
                    "Prediksi Sedang",
                    "Prediksi Rendah"
                ]
            )


            st.dataframe(
                cm_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # Fitur model
            # ------------------------------------------------

            st.markdown(
                "### ⚙️ Fitur yang Digunakan Model"
            )


            for nomor, nama_fitur in enumerate(
                fitur,
                start=1
            ):

                st.write(
                    f"{nomor}. `{nama_fitur}`"
                )


            st.info(
                """
                **Catatan penting:**

                Rasio kehadiran tidak digunakan sebagai
                fitur Naive Bayes karena rasio tersebut
                digunakan untuk membentuk label kedisiplinan.

                Dengan demikian, model tidak diberikan
                jawaban secara langsung (data leakage).
                """
            )


# ============================================================
# 21. TENTANG APLIKASI
# ============================================================

if menu == "ℹ️ Tentang Aplikasi":

    st.markdown(
        '<div class="section-title">ℹ️ Tentang Aplikasi</div>',
        unsafe_allow_html=True
    )


    with st.expander(
        "📖 Tentang Aplikasi",
        expanded=True
    ):

        st.write(
            """
            Aplikasi ini merupakan sistem bantu analisis
            untuk mengintegrasikan data Fingerprint dan
            Buku Kegiatan Mandor (BKM) dalam melakukan
            klasifikasi tingkat kedisiplinan karyawan.
            """
        )


    with st.expander(
        "🔄 Alur Pengolahan Data"
    ):

        st.write(
            """
            1. Pengguna mengunggah dua file Excel.

            2. Sistem membaca struktur data.

            3. Sistem membaca tanggal pada kolom BKM Date.

            4. Sistem menentukan periode data secara otomatis.

            5. Data dari kedua file digabungkan.

            6. Data Fingerprint dan BKM diolah menjadi
               variabel penelitian.

            7. Rasio kehadiran dihitung.

            8. Kategori kedisiplinan dibentuk berdasarkan
               rasio kehadiran.

            9. Fitur Fingerprint dan BKM digunakan
               untuk proses Naive Bayes.

            10. Data dibagi menjadi data training dan
                data testing.

            11. Model Gaussian Naive Bayes dilatih.

            12. Model menghasilkan prediksi klasifikasi.

            13. Model dievaluasi menggunakan Accuracy,
                Precision, Recall, F1-Score dan
                Confusion Matrix.
            """
        )


    with st.expander(
        "📊 Kriteria Kedisiplinan"
    ):

        st.write(
            """
            Kriteria yang digunakan dalam penelitian:

            🟢 Tinggi
            = Rasio Kehadiran ≥ 90%

            🟡 Sedang
            = Rasio Kehadiran 75% sampai < 90%

            🔴 Rendah
            = Rasio Kehadiran < 75%

            Kriteria ini merupakan label penelitian
            yang dibentuk berdasarkan rasio kehadiran,
            bukan kategori resmi perusahaan.
            """
        )


    with st.expander(
        "📁 Format Data"
    ):

        st.write(
            """
            Kedua file Excel sebaiknya memiliki kolom
            yang berkaitan dengan:

            • Employee
            • Employee Name
            • BKM Date
            • Time In
            • Time Out
            • BKM Number
            • Act Code
            • Manday
            • Output
            • Premi
            • Penalty
            • Overtime
            • 1-Valid 0-Non

            Nama beberapa kolom masih dapat dikenali
            menggunakan variasi nama yang umum.
            """
        )


# ============================================================
# 22. DOWNLOAD HASIL
# ============================================================

if dataset is not None:

    st.divider()

    st.markdown(
        "## 📥 Download Hasil"
    )


    # --------------------------------------------------------
    # Buat Excel
    # --------------------------------------------------------

    output = io.BytesIO()


    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataset.to_excel(
            writer,
            sheet_name="Hasil Klasifikasi",
            index=False
        )


        if evaluasi is not None:

            evaluasi_tabel = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score",
                        "Jumlah Data Training",
                        "Jumlah Data Testing"
                    ],

                    "Nilai": [
                        evaluasi["Accuracy"],
                        evaluasi["Precision"],
                        evaluasi["Recall"],
                        evaluasi["F1-Score"],
                        evaluasi[
                            "Jumlah Data Training"
                        ],
                        evaluasi[
                            "Jumlah Data Testing"
                        ]
                    ]
                }
            )


            evaluasi_tabel.to_excel(
                writer,
                sheet_name="Evaluasi Model",
                index=False
            )


        if data_gabungan is not None:

            data_gabungan.to_excel(
                writer,
                sheet_name="Data Gabungan",
                index=False
            )


    output.seek(0)


    st.download_button(

        label="📥 Download Hasil Klasifikasi Excel",

        data=output,

        file_name="hasil_klasifikasi_kedisiplinan.xlsx",

        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        use_container_width=True
    )


# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.caption(
    "Sistem Klasifikasi Tingkat Kedisiplinan Karyawan "
    " | Integrasi Fingerprint + BKM | Naive Bayes"
)