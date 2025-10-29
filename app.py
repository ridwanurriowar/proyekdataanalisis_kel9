import streamlit as st
import joblib
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np


st.title("Prediksi Volume Produksi Ikan Pembenihan di Jawa Barat")
st.write("Aplikasi ini memprediksi volume produksi ikan pembenihan berdasarkan jenis ikan dan kabupaten/kota di Jawa Barat.")

st.markdown("""
### Cara Menggunakan Aplikasi

1.  Pilih **Kelompok Ikan** yang ingin Anda prediksi volumenya dari dropdown menu.
2.  Pilih **Kabupaten / Kota** tempat produksi ikan dari dropdown menu.
3.  Masukkan jumlah tahun ke depan yang ingin Anda prediksi volumenya.
4.  Masukkan nilai **Nilai (Rp. Juta)** dan **Harga Rata-Rata Tertimbang (Rp/ ribu ekor)** yang diperkirakan untuk setiap tahun prediksi di kolom input yang tersedia.
5.  Klik tombol **"Prediksi Volume"** untuk mendapatkan hasil prediksi.
6.  Hasil prediksi volume produksi untuk tahun berikutnya akan ditampilkan di bawah tombol.
7.  Sebuah plot yang membandingkan data historis dengan hasil prediksi juga akan ditampilkan.
""")

# --- FITUR UNGGAH DATA ---
st.sidebar.title("⚙️ Opsi Data")
uploaded_file = st.sidebar.file_uploader(
    "Unggah Dataset Kustom (Opsional)",
    type=["xlsx"],
    help="Unggah file Excel (.xlsx) dengan kolom yang sama dengan dataset default: 'Kelompok Ikan', 'Kab / Kota', 'Tahun', 'Volume (Ribu Ekor)', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'"
)

df = None  # Inisialisasi dataframe
default_data_path = './content/produksi_pembenihan_jawaBarat_2019_2023_filtered.xlsx'

if uploaded_file is not None:
    st.sidebar.success("Menggunakan file yang diunggah.")
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file yang diunggah: {e}")
        st.stop()
else:
    st.sidebar.info("Menggunakan dataset default.")
    try:
        df = pd.read_excel(default_data_path)
    except FileNotFoundError:
        st.error(f"Error: File data default tidak ditemukan di '{default_data_path}'. Harap unggah file kustom di sidebar untuk melanjutkan.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file default: {e}")
        st.stop()

# --- VALIDASI DATASET ---
if df is not None:
    required_columns = [
        'Kelompok Ikan', 
        'Kab / Kota', 
        'Tahun', 
        'Volume (Ribu Ekor)', 
        'Nilai (Rp. Juta)', 
        'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'
    ]
    
    # Periksa apakah semua kolom yang diperlukan ada
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"Dataset yang dimuat tidak memiliki kolom yang diperlukan: {', '.join(missing_cols)}")
        st.stop()
else:
    st.error("Dataframe tidak berhasil dimuat. Aplikasi berhenti.")
    st.stop()

# --- LANJUTAN APLIKASI (KODE ASLI ANDA) ---

# Buat tabel jenis ikan per kota
fish_production_by_city = df.groupby('Kab / Kota')['Kelompok Ikan'].unique().reset_index()
fish_production_by_city['Kelompok Ikan'] = fish_production_by_city['Kelompok Ikan'].apply(lambda x: ', '.join(x))

st.subheader("Jenis Ikan yang Diproduksi per Kabupaten / Kota:")
st.table(fish_production_by_city)

# Dapatkan daftar unik dari dataframe yang dimuat
fish_groups = df['Kelompok Ikan'].unique().tolist()
cities = df['Kab / Kota'].unique().tolist()

selected_fish_group = st.selectbox("Pilih Kelompok Ikan:", fish_groups)
selected_city = st.selectbox("Pilih Kabupaten / Kota:", cities)

# Input untuk jumlah tahun prediksi
num_future_years = st.number_input("Jumlah tahun ke depan untuk prediksi:", min_value=1, value=3, step=1)

# Input untuk nilai regressor masa depan
future_regressor_values = {}
st.write(f"Masukkan nilai regressor untuk {num_future_years} tahun ke depan:")

# Dapatkan tahun historis terakhir dari dataframe
last_historical_year = df['Tahun'].max()

for i in range(num_future_years):
    year = last_historical_year + 1 + i
    st.write(f"Tahun: {year}")
    future_nilai = st.number_input(f"Nilai (Rp. Juta) untuk {year}:", min_value=1, value=int(df['Nilai (Rp. Juta)'].mean()), key=f'nilai_{year}')
    future_harga = st.number_input(f"Harga Rata-Rata Tertimbang (Rp/ ribu ekor) untuk {year}:", min_value=1, value=int(df['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].mean()), key=f'harga_{year}')
    future_regressor_values[year] = {'Nilai (Rp. Juta)': future_nilai, 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': future_harga}


if st.button("Prediksi Volume"):
    # Buat nama file model
    model_filename = f'prophet_model_{selected_fish_group.replace(" ", "_")}_{selected_city.replace(" ", "_")}.pkl'
    model_filepath = os.path.join('./content/prophet_models', model_filename)

    if not os.path.exists(model_filepath):
        st.warning(f"Model untuk {selected_fish_group} di {selected_city} tidak ditemukan.")
    else:
        # Muat model yang telah dilatih
        model = joblib.load(model_filepath)

        # Buat dataframe masa depan untuk prediksi
        future = model.make_future_dataframe(periods=num_future_years, freq='Y') # Prediksi untuk jumlah tahun ke depan
        future['ds'] = future['ds'].dt.tz_localize(None) # Hapus informasi zona waktu jika ada

        # Dapatkan tanggal historis terakhir dari data pelatihan
        last_historical_date = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)]['Tahun'].max()
        last_historical_date = pd.to_datetime(last_historical_date, format="%Y")

        # Dapatkan nilai regressor historis
        historical_regressors = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)].copy()
        historical_regressors['ds'] = pd.to_datetime(historical_regressors['Tahun'], format="%Y")
        historical_regressors = historical_regressors[['ds', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']]

        # Gabungkan nilai regressor historis ke dataframe masa depan
        future = future.merge(historical_regressors, on='ds', how='left')

        # Tetapkan nilai regressor masa depan yang diinput pengguna
        for year, values in future_regressor_values.items():
            future.loc[future['ds'].dt.year == year, 'Nilai (Rp. Juta)'] = values['Nilai (Rp. Juta)']
            future.loc[future['ds'].dt.year == year, 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'] = values['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']

        # Isi nilai NaN di kolom regressor (untuk tanggal historis)
        future['Nilai (Rp. Juta)'] = future['Nilai (Rp. Juta)'].fillna(method='ffill')
        future['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'] = future['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].fillna(method='ffill')

        # Lakukan prediksi
        forecast = model.predict(future)

        # Tampilkan prediksi untuk tahun-tahun mendatang
        future_forecast = forecast[forecast['ds'] > last_historical_date]
        if not future_forecast.empty:
            st.subheader("Prediksi Volume Produksi untuk Tahun Berikutnya:")
            for index, row in future_forecast.iterrows():
                st.write(f"Tahun: {row['ds'].year}, Volume (Ribu Ekor): {row['yhat']:.2f}")
        else:
            st.write("Tidak ada prediksi untuk tahun mendatang.")

        # Tampilkan plot historis vs prediksi
        st.subheader("Historis vs Prediksi")
        historical_subset = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)].copy()
        historical_subset['ds'] = pd.to_datetime(historical_subset['Tahun'], format="%Y")
        # Agregasi jika ada duplikat tahun
        historical_subset = historical_subset.groupby('ds', as_index=False).agg({'Volume (Ribu Ekor)': 'sum'}).rename(columns={'Volume (Ribu Ekor)': 'y'})


        plt.figure(figsize=(10, 6))
        plt.plot(historical_subset["ds"], historical_subset["y"], label="Aktual", marker="o")
        plt.plot(forecast["ds"], forecast["yhat"], label="Prediksi", marker="x")
        plt.legend()
        plt.title(f"{selected_fish_group} - {selected_city}: Aktual vs Prediksi")
        plt.xlabel("Tahun")
        plt.ylabel("Volume (Ribu Ekor)")
        plt.grid(True)
        st.pyplot(plt)

st.markdown("""
### Informasi Model

Model prediksi yang digunakan dalam aplikasi ini adalah **Prophet**, sebuah *forecasting procedure* yang dikembangkan oleh Facebook. Model ini cocok untuk data deret waktu dengan efek musiman yang kuat dan tren dari tahun ke tahun.

Model ini juga menggunakan **regressor tambahan** yaitu 'Nilai (Rp. Juta)' dan 'Harga Rata-Rata Tertimbang (Rp/ ribu ekor)'. Penggunaan regressor tambahan ini bertujuan untuk meningkatkan akurasi prediksi dengan mempertimbangkan faktor-faktor lain yang mungkin mempengaruhi volume produksi.

**Disclaimer:** Prediksi ini didasarkan pada data historis dan nilai regressor yang Anda masukkan. Hasil prediksi merupakan estimasi dan tidak dapat dijamin 100% akurat. Berbagai faktor eksternal yang tidak termasuk dalam model dapat mempengaruhi hasil aktual.
""")
