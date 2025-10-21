import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import pydeck as pdk

# Konfigurasi Halaman 
st.set_page_config(page_title="Prediksi Risiko Stunting", layout="wide")
st.title("Prediksi Risiko Stunting di Sulawesi Tenggara")
st.markdown("""
Aplikasi ini memprediksi **tingkat risiko stunting** berdasarkan faktor gizi dan sosial  
menggunakan **model K-Nearest Neighbors (KNN)**.
""")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data_cleaned.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Data Koordinat Kabupaten/Kota 
@st.cache_data
def load_kabupaten_coords():
    data = {
        "Kabupaten_Kota": [
            "Kendari", "Baubau", "Konawe", "Konawe Selatan", "Konawe Kepulauan",
            "Konawe Utara", "Buton", "Buton Selatan", "Buton Utara", "Buton Tengah",
            "Muna", "Muna Barat", "Kolaka", "Kolaka Utara", "Kolaka Timur",
            "Bombana", "Wakatobi"
        ],
        "Latitude": [
            -3.9747, -5.5063, -3.9593, -4.4146, -4.1857, -3.3892,
            -5.0698, -5.4796, -4.6972, -5.4597, -5.0298, -4.7962,
            -4.2388, -3.2409, -3.6776, -4.8943, -5.7602
        ],
        "Longitude": [
            122.5142, 122.5853, 119.2208, 121.7233, 122.9530, 121.7925,
            123.1726, 121.7004, 122.6704, 122.1188, 122.2386, 122.4933,
            121.4245, 120.5916, 120.8802, 121.4091, 123.7304
        ]
    }
    return pd.DataFrame(data)

df_coords = load_kabupaten_coords()

# Pilih fitur & target 
selected_features = [
    'Jumlah Balita Pendek (TB/U)',
    'Jumlah Balita Gizi Buruk (BB/TB : < -3 SD)',
    'Jumlah Balita Gizi Kurang (BB/TB : < -2 sd -3 SD)',
    'Persentase Penduduk Miskin',
    'wasting',
    'Rumah Tangga yang Memiliki Akses Terhadap Sanitasi Layak',
    'Bayi Baru Lahir (Jumlah Mendapat IMD)',
    'Jumlah Balita Yang Diukur Tinggi Badan',
    'Jumlah Balita Yang Ditimbang',
    'Bayi Usia <6 Bulan (Jumlah Diberi ASI Eksklusif)',
    'Perempuan',
    'Garis Kemiskinan'
]

X = df[selected_features]
y = df['stunting']

# Label biner berdasarkan median
threshold = y.median()
y_class = np.where(y > threshold, "Tinggi", "Rendah")

# Load Scaler & Model dari File 
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# Evaluasi Akurasi Cepat 
try:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y_class, y_pred)
except Exception:
    acc = np.nan

# Input User 
st.sidebar.header("Masukkan Data Faktor Sosial & Gizi (10 Wilayah)")

decimal_features = [
    'Persentase Penduduk Miskin',
    'wasting',
    'Rumah Tangga yang Memiliki Akses Terhadap Sanitasi Layak',
    'Bayi Usia <6 Bulan (Jumlah Diberi ASI Eksklusif)',
    'Perempuan'
]

input_list = []
for i in range(1, 10 + 1):
    st.sidebar.subheader(f"Wilayah {i}")
    wilayah = st.sidebar.selectbox(
        f"Nama Kabupaten/Kota {i}", df_coords["Kabupaten_Kota"], key=f"wilayah_{i}"
    )
    wilayah_data = {"Kabupaten_Kota": wilayah}

    for feature in selected_features:
        median_val = float(np.median(df[feature].dropna()))
        fmt = "%.2f" if feature in decimal_features else "%.0f"
        wilayah_data[feature] = st.sidebar.number_input(
            label=f"{feature} ({wilayah})",
            value=median_val,
            format=fmt,
            key=f"{feature}_{i}"
        )
    input_list.append(wilayah_data)

input_df = pd.DataFrame(input_list)

# Standarisasi & Prediksi
try:
    input_scaled = scaler.transform(input_df[selected_features])
    predictions = model.predict(input_scaled)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)
        input_df["Probabilitas Tinggi"] = probs[:, 1]
        input_df["Probabilitas Rendah"] = probs[:, 0]
    else:
        input_df["Probabilitas Tinggi"] = np.nan
        input_df["Probabilitas Rendah"] = np.nan

    input_df["Prediksi Risiko"] = predictions
except Exception as e:
    st.error(f"Gagal melakukan prediksi: {e}")
    st.stop()

# Gabung dengan koordinat untuk peta 
df_map = pd.merge(df_coords, input_df, on="Kabupaten_Kota", how="left")
df_map["color"] = df_map["Prediksi Risiko"].map(
    {"Tinggi": [255, 0, 0], "Rendah": [0, 200, 0]}
)
df_map["color"] = df_map["color"].apply(lambda x: x if isinstance(x, list) else [200, 200, 200])

# Peta Interaktif 
st.markdown("---")
st.subheader("Peta Risiko Stunting di Sulawesi Tenggara")

view_state = pdk.ViewState(
    latitude=df_map["Latitude"].mean(),
    longitude=df_map["Longitude"].mean(),
    zoom=6.5
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=["Longitude", "Latitude"],
    get_color="color",
    get_radius=7000,
    pickable=True
)

tooltip = {"text": "{Kabupaten_Kota}\nRisiko: {Prediksi Risiko}"}
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

#  Bar Chart 
st.markdown("---")
st.subheader("Perbandingan Risiko Stunting Wilayah Input")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=input_df, x="Kabupaten_Kota", y="Probabilitas Tinggi", palette="Reds", ax=ax)
ax.set_ylabel("Probabilitas Risiko Tinggi")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig)

# Heatmap Faktor
st.markdown("---")
st.subheader("Heatmap Faktor Sosial & Gizi (Input Pengguna)")

numeric_input = input_df[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
scaler_minmax = MinMaxScaler()
scaled_values = scaler_minmax.fit_transform(numeric_input)
scaled_df = pd.DataFrame(scaled_values, columns=selected_features)
scaled_df["Kabupaten_Kota"] = input_df["Kabupaten_Kota"]

n_cols = 2
n_rows = int(np.ceil(len(scaled_df) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()

for i, (idx, row) in enumerate(scaled_df.iterrows()):
    data = pd.DataFrame(row[selected_features].astype(float)).T
    sns.heatmap(
        data,
        annot=True,
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        vmin=0, vmax=1,
        ax=axes[i]
    )
    axes[i].set_title(f"{row['Kabupaten_Kota']}", fontsize=11)
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
> **Catatan Penggunaan:**
> - Masukkan **angka normal** untuk jumlah (contoh: `657`)  
> - Masukkan **dua desimal** untuk persentase dan kolom *Perempuan* (contoh: `7.45`)  
> - **Garis Kemiskinan** tidak pakai desimal (.00)  
""")
