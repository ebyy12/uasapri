# Laporan Proyek Machine Learning

### Nama : Desby Permata Sulaeman
### Nim : 211351042
### Kelas : Pagi B

## Domain Proyek

Prediksi serangan jantung adalah upaya untuk mengidentifikasi faktor-faktor risiko yang dapat meningkatkan kemungkinan seseorang mengalami serangan jantung di masa depan. Serangan jantung, juga dikenal sebagai infark miokard atau penyakit jantung koroner, terjadi ketika pasokan darah ke otot jantung terhenti atau berkurang secara signifikan, biasanya akibat penyumbatan arteri koroner. Prediksi serangan jantung melibatkan penilaian berbagai faktor risiko yang dapat meningkatkan kemungkinan seseorang mengembangkan penyakit jantung koroner.

## Business Understanding

Untuk orang-orang yang ingin mengetahui sekumpulan data analisis dan prediksi serangan jantung.

Bagian Laporan ini mencakup:

### Problem Statements

Ketidakmungkinan bagi seseorang untuk mengetahui prediksi serangan jantung tanpa memprediksinya terlebih dahulu.

### Goals

- Membuat penelitian dalam studi prediksi serangan jantung.

-  Membangun model prediktif berbasis logistik regresi yang memanfaatkan dataset "Heart Attack Analysis & Prediction" untuk mengidentifikasi faktor-faktor risiko utama yang berkontribusi terhadap serangan jantung.

### Solution Statements

- Pengembangan platform pencarian kumpulan data prediksi dalam Heart Attack Analysis & Prediction Dataset yang menintegrasikan data dari kaggle.com untuk memberikan pengguna akses cepat dan mudah ke informasi tentang Kumpulan Data Analisis dan Prediksi Serangan Jantung. Platform ini akan menyediakan antarmuka pengguna yang ramah.

- Model yang dihasilkan dari dataset itu menggunakan metode Logistics Regression.
    

## Data Understanding

Dataset yang saya gunakan berasal dari Kaggle yang berisi kumpulan data analisis dan prediksi serangan jantung. Dataset ini mengandung 303 baris dan memiliki 14 kolom setelah dilakukan pengecekan.

[Heart Attack Analysis & Prediction]
https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- age       : Menunjukan usia pasien. [Tipe data: Int]
- sex       : Menunjukan jenis kelamin pasien. [Tipe data: Int]
- cp        : Menunjukan tipe nyeri dada tipe nyeri dada. [Tipe data: Int]
- trtbps    : Menunjukan tekanan darah istirahat (dalam mm Hg). [Tipe data: Int]
- chol      : Menunjukan kolesterol dalam mg/dl diambil melalui sensor BMI. [Tipe data: Int]
- fbs       : (Menunjukan gula darah puasa>120 mg/dl) (1=benar; 0;salah). [Tipe data: Int]
- restecg   : Menunjukan hasil elektrokardiografi istirahat. [Tipe data: Int]
- thalachh  : Menunjukan detak jantung maksimum tercapai. [Tipe data: Int]
- exng      :Menunjukan angina akibat olahraga (1 = ya;0 = tidak). [Tipe data: Int]
- oldpeak   : Menunjukan puncak sebelumnya. [Tipe data: Float]
- slp       : Menunjukan Lereng. [Tipe data: Int]
- caa       : Menunjukan jumlah kapal besar (0-3). [Tipe data: Int]
- thall     : Menunjukan tarifnya. [Tipe data: Int]
- output    : Menunjukan variabel sasaran. [Tipe data: Int]

## Data Preparation

## Data Collection

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Heart Attack Analysis & Prediction Dataset jika Anda tertarik dengan datasetnya, Anda bisa click link diatas.

## Data Discovery And Profilling

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan,

``` bash 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
``` 

- Karena kita menggunakan vscode untuk mengerjakan maka file lokal harus berada di direktori yang sama,

``` bash 
df = pd.read_csv('heart.csv')
df.head()
``` 

- Lalu tipe data dari masing-masing kolom, kita bisa menggunakan properti info,

``` bash 
df.info()
```


- Selanjutnya kita akan memeriksa apakah dataset tersebut terdapat baris yang kosong atau null dengan menggunakan seaborn,

``` bash 
sns.heatmap(df.isnull())
```
![Alt text](output.png) <br>

- Selanjutnya mari kita lanjutkan dengan data exploration,

``` bash 
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```
![Alt text](output2.png) <br>

- Mari lanjutkan dengan modeling.

## Modeling

Model regresi logistik adalah sebuah jenis model statistik yang digunakan untuk menganalisis hubungan antara satu atau lebih variabel independen (prediktor) dengan variabel dependen biner (dua kategori, seperti ya/tidak, sukses/gagal, atau positif/negatif).

- Sebelumnya mari kita import library yang nanti akan digunakan,

``` bash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

- Langkah pertama adalah memasukan kolom-kolom fitur yang ada di datasets dan juga kolom targetnya,

``` bash
X = df.drop (columns='output', axis=1)
Y = df['output']
```

- Pembagian X dan Y menjadi train dan testnya masing-masing,
``` bash
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=2)
```

- Mari kita lanjut dengan membuat model,

``` bash
model = LogisticRegression()
```

- Mari lanjut, memasukkan x_train dan y_train pada model dan memasukan X_train_pred,

``` bash
model.fit(X_train, Y_train)
X_train_pred = model.predict(X_train)
```

- Sekarang kita bisa melihat akurasi dari model kita,

``` bash
training_data_accuracy = accuracy_score(X_train_pred, Y_train)
print('akurasi : ', training_data_accuracy)
```

- Akurasi modelnya yaitu 83%, selanjutnya mari kita test menggunakan sebuah array value, 

``` bash
input_data = (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)
input_data_np = np.array(input_data)
input_data_reshape = input_data_np.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

prediksi = model.predict(std_data)
print(prediksi)

if(prediksi[0] == 1 ):
    print("Positif")
else:
    print('Negatif')
```

- Sekarang modelnya sudah selesai, mari kita export sebagai file sav agar nanti bisa kita gunakan pada project web streamlit kita,

``` bash
import pickle

filename = 'PrediksiHeart.sav'
pickle.dump(model, open(filename,'wb'))
```

- Mari lanjut, memasukkan x_train dan y_train pada model dan memasukkan value predict pada y_pred,

``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=2)
```

## Evaluation

Metrik yang digunakan yaitu metrik akurasi.

- Matriks Akurasi (Accuracy Matrix) yang merupakan salah satu metode evaluasi yang digunakan dalam konteks klasifikasi (classification) untuk mengukur sejauh mana model klasifikasi mampu memprediksi dengan benar.

## Library evaluasi
``` bash
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Compute performance manually
NewprediksiBenar = (predicted == Y_train).sum()
NewprediksiSalah = (predicted != Y_train).sum()

print("prediksi benar: ", NewprediksiBenar, " data")
print("prediksi salah: ", NewprediksiSalah, " data")
print("Akurasi Algoritme: ", NewprediksiBenar/(NewprediksiBenar+NewprediksiSalah)*100,"%") 
```

## Deployment

https://github.com/ebyy12/Heart_Attack

https://heartattack-dx6vbyqjvpt8b2zvspmtac.streamlit.app/
