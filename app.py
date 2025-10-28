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


# Load the original dataframe to get unique values for dropdowns
# Assuming the filtered dataframe is saved as 'produksi_pembenihan_jawaBarat_2019_2023_filtered.xlsx'
try:
    df = pd.read_excel('./content/produksi_pembenihan_jawaBarat_2019_2023_filtered.xlsx')
except FileNotFoundError:
    st.error("Error: Filtered data file not found. Please ensure 'produksi_pembenihan_jawaBarat_2019_2023_filtered.xlsx' is in the correct directory.")
    st.stop()


fish_groups = df['Kelompok Ikan'].unique().tolist()
cities = df['Kab / Kota'].unique().tolist()

selected_fish_group = st.selectbox("Pilih Kelompok Ikan:", fish_groups)
selected_city = st.selectbox("Pilih Kabupaten / Kota:", cities)

# Input for number of future years to predict
num_future_years = st.number_input("Jumlah tahun ke depan untuk prediksi:", min_value=1, value=3, step=1)

# Input for future regressor values for multiple years
future_regressor_values = {}
st.write(f"Masukkan nilai regressor untuk {num_future_years} tahun ke depan:")

# Get the last historical year
last_historical_year = df['Tahun'].max()

for i in range(num_future_years):
    year = last_historical_year + 1 + i
    st.write(f"Tahun: {year}")
    future_nilai = st.number_input(f"Nilai (Rp. Juta) untuk {year}:", min_value=0.0, value=float(df['Nilai (Rp. Juta)'].mean()), key=f'nilai_{year}')
    future_harga = st.number_input(f"Harga Rata-Rata Tertimbang (Rp/ ribu ekor) untuk {year}:", min_value=0.0, value=float(df['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].mean()), key=f'harga_{year}')
    future_regressor_values[year] = {'Nilai (Rp. Juta)': future_nilai, 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': future_harga}


if st.button("Prediksi Volume"):
    # Construct the model filename
    model_filename = f'prophet_model_{selected_fish_group.replace(" ", "_")}_{selected_city.replace(" ", "_")}.pkl'
    model_filepath = os.path.join('./content/prophet_models', model_filename)

    if not os.path.exists(model_filepath):
        st.warning(f"Model for {selected_fish_group} in {selected_city} not found.")
    else:
        # Load the trained model
        model = joblib.load(model_filepath)

        # Create future dataframe for prediction
        future = model.make_future_dataframe(periods=num_future_years, freq='Y') # Predict for the specified number of future years
        future['ds'] = future['ds'].dt.tz_localize(None) # Remove timezone information if present

        # Get the last historical date from the training data for this combination
        last_historical_date = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)]['Tahun'].max()
        last_historical_date = pd.to_datetime(last_historical_date, format="%Y")

        # Get the historical regressor values
        historical_regressors = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)].copy()
        historical_regressors['ds'] = pd.to_datetime(historical_regressors['Tahun'], format="%Y")
        historical_regressors = historical_regressors[['ds', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']]

        # Merge historical regressor values onto the future dataframe for historical dates
        future = future.merge(historical_regressors, on='ds', how='left')

        # Assign the user-input future regressor values to the future dates
        for year, values in future_regressor_values.items():
            future.loc[future['ds'].dt.year == year, 'Nilai (Rp. Juta)'] = values['Nilai (Rp. Juta)']
            future.loc[future['ds'].dt.year == year, 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'] = values['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']

        # Fill NaN values in regressor columns in the future dataframe with the last known value for historical dates
        future['Nilai (Rp. Juta)'] = future['Nilai (Rp. Juta)'].fillna(method='ffill')
        future['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'] = future['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].fillna(method='ffill')

        # Make prediction
        forecast = model.predict(future)

        # Display the prediction for the future years
        future_forecast = forecast[forecast['ds'] > last_historical_date]
        if not future_forecast.empty:
            st.subheader("Prediksi Volume Produksi untuk Tahun Berikutnya:")
            for index, row in future_forecast.iterrows():
                 st.write(f"Tahun: {row['ds'].year}, Volume (Ribu Ekor): {row['yhat']:.2f}")

        else:
            st.write("Tidak ada prediksi untuk tahun mendatang.")

        # Optional: Display historical vs forecast plot
        st.subheader("Historis vs Prediksi")
        historical_subset = df[(df['Kelompok Ikan'] == selected_fish_group) & (df['Kab / Kota'] == selected_city)].copy()
        historical_subset['ds'] = pd.to_datetime(historical_subset['Tahun'], format="%Y")
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
