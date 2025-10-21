# Prediksi Resiko Stunting di Sulawesi Tenggara Berdasarkan Faktor Gizi dan Sosial Menggunakan Algoritma K-Nearest Neighbors Classifier (KNN)

Aplikasi ini digunakan untuk memprediksi risiko stunting pada balita di suatu wilayah, dengan menampilkan apakah wilayah tersebut tergolong risiko tinggi atau risiko rendah terhadap stunting.

## Tim Pengembang
Kelompok 2  
- Nirmala (E1E123012)  
- Rahmah Yuniati (E1E123074)  
- Muh. Arif Rahman Gani (E1E123040)

##  Deskripsi Proyek
Stunting merupakan kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis.  
Proyek ini bertujuan untuk:

- Menganalisis faktor-faktor yang mempengaruhi prevalensi stunting di Sulawesi Tenggara  
- Membangun model prediksi risiko stunting menggunakan algoritma *Machine Learning (KNN)*  
- Menyediakan visualisasi interaktif untuk membantu pengambilan keputusan kebijakan kesehatan  

##  Pertanyaan Penelitian
1. Bagaimana pengaruh tingkat akses terhadap sanitasi layak terhadap prevalensi stunting?  
2. Bagaimana hubungan antara persentase bayi yang mendapatkan ASI eksklusif dengan penurunan prevalensi stunting?  
3. Bagaimana pengaruh tingkat kemiskinan terhadap prevalensi stunting?  
4. Bagaimana hubungan antara rata-rata lama sekolah perempuan dengan tingkat stunting pada balita?  
5. Bagaimana sebaran data dan hubungan antar indikator gizi, sanitasi, dan sosial ekonomi terhadap kejadian stunting di wilayah Sulawesi Tenggara?  

##  Teknologi yang Digunakan

**Machine Learning & Data Processing**
- Python 3.8+
- scikit-learn — algoritma KNN dan preprocessing  
- pandas — manipulasi data  
- numpy — komputasi numerik  
- joblib — penyimpanan model 


**Visualisasi & Web App**
- Streamlit — framework aplikasi web  
- matplotlib — plotting data  
- seaborn — visualisasi statistik  
- pydeck — peta interaktif 3D  

**Development Tools**
- Google Colab
- GitHub 

## Struktur Proyek
prediksi-stunting-sultra/
│
├── app.py # Aplikasi utama Streamlit
├── data_cleaned.csv # Dataset yang telah dibersihkan
├── model.pkl # Model KNN terlatih
├── scaler.pkl # Scaler untuk normalisasi data
├── requirements.txt # Dependencies Python
├── README.md # Dokumentasi proyek
└── notebooks/
└── stunting_analysis.ipynb # Notebook analisis & training

## Metodologi
1. Data Wrangling

Tahap ini bertujuan untuk menyiapkan data mentah agar bersih, konsisten, dan siap dianalisis. Prosesnya meliputi:

a. Data Gathering

Mengumpulkan dataset dari website pemerintah terkait stunting dan faktor-faktornya di Sulawesi Tenggara:

- Akses Sanitasi Layak (.csv)

- Pemberian ASI Eksklusif (.xlsx)

- Status Gizi Balita (.xlsx)

- Jumlah dan Persentase Penduduk Miskin (.csv)

- Rata-rata Lama Sekolah (.csv)

Data ini diambil dari berbagai format file (CSV, Excel) dan diimpor ke Python menggunakan pandas.

b. Data Assessing

- Memeriksa struktur dataset (.info()), untuk melihat jumlah baris, kolom, tipe data, dan kelengkapan informasi.

- Mengidentifikasi missing values (.isna().sum()) untuk mengetahui kolom yang perlu penanganan.

- Mengecek duplikasi (.duplicated().sum()), agar data tidak terhitung ganda.

- Melakukan analisis statistik deskriptif (.describe()), untuk memahami sebaran dan distribusi numerik.

c. Data Cleaning

- Menangani outlier menggunakan metode winsorization, agar nilai ekstrim tidak memengaruhi analisis:

- Dilakukan pada variabel numerik yang relevan di setiap dataset.

- Menyesuaikan nama kolom agar konsisten antar dataset, termasuk:

- Mengubah nama kolom yang panjang atau tidak standar (rename_mapping).

- Menyamakan kolom Kabupaten/Kota menjadi Kabupaten_Kota di semua dataset.

- Menangani kolom yang hilang atau salah format jika ada, agar mudah untuk proses merge dan analisis berikutnya.

d. Feature Engineering

- Memilih fitur yang relevan berdasarkan korelasi dengan target stunting:

- Menghitung korelasi antar variabel numerik (.corr()).

- Menentukan threshold korelasi (≥ 0.3) untuk memilih fitur prediktor penting.

- Fitur yang dipilih meliputi indikator gizi, sanitasi, pendidikan, kemiskinan, dan ASI eksklusif.

e. Merge Data

- Menggabungkan semua dataset menjadi satu dataframe komprehensif (pd.merge()), menggunakan kolom Kabupaten_Kota sebagai kunci.

- Merge dilakukan secara outer join, sehingga semua kabupaten/kota tercakup meskipun ada data yang hilang di beberapa sumber.

2. Data Preprocessing

Tahap ini menyiapkan data agar siap digunakan oleh model machine learning:

- Standarisasi fitur numerik menggunakan StandardScaler, agar semua variabel berada pada skala yang sama.

- Klasifikasi target stunting menjadi dua kategori:

  - Rendah (≤ median)

  - Tinggi (> median)

- Split data menjadi training dan testing (train_test_split) dengan stratifikasi, agar proporsi kelas tetap seimbang.

3. Exploratory Data Analysis (EDA)

- Visualisasi hubungan antar variabel:
Scatter plot antar indikator gizi, sanitasi, kemiskinan, pendidikan, dan ASI terhadap stunting.

- Heatmap korelasi antar variabel untuk memahami hubungan linear antar fitur.

- Pairplot untuk melihat sebaran dan hubungan multivariat antar fitur setelah standarisasi.

- Memberikan insight awal mengenai faktor-faktor yang paling memengaruhi stunting di tiap kabupaten/kota.

4. Modelling

- Algoritma yang digunakan: K-Nearest Neighbors (KNN) untuk klasifikasi risiko stunting.

- Parameter model: n_neighbors=3.

- Evaluasi model:

  -Confusion Matrix

  -Classification Report (Precision, Recall, F1-score)

  -Akurasi model

5. Penyimpanan Model

- Model KNN dan objek scaler disimpan menggunakan joblib untuk digunakan kembali dalam deployment atau aplikasi prediksi:

  - model.pkl → model KNN

  - scaler.pkl → StandardScaler

## Hasil dan Kesimpulan
🔹 Temuan Utama

1. Faktor Pendidikan: Rata-rata lama sekolah perempuan berkorelasi negatif dengan stunting.

2. Kemiskinan: Tingkat kemiskinan memiliki pengaruh signifikan terhadap prevalensi stunting.

3. Sanitasi: Akses terhadap sanitasi layak berperan penting dalam pencegahan stunting.

4. ASI Eksklusif: Pemberian ASI eksklusif berkontribusi pada penurunan risiko stunting.

🔹 Intervensi yang disarankan mencakup:

- Peningkatan pendidikan perempuan

- Pengentasan kemiskinan

- Akses sanitasi dan air bersih

- Promosi ASI eksklusif

## Dependencies
Lihat file requirements.txt untuk daftar lengkap.
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- seaborn>=0.12.0
- matplotlib>=3.7.0
- joblib>=1.3.0
- pydeck>=0.8.0

## Lisensi
Proyek ini dibuat untuk keperluan akademis dan penelitian.
