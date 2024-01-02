# Laporan Proyek Machine Learning

### Nama : Desby Permata Sulaeman
### Nim : 211351042
### Kelas : Pagi B

## Domain Proyek
Apriori adalah algoritma untuk menambang itemset yang sering muncul dan pembelajaran aturan asosiasi pada basis data relasional. Algoritma ini berlanjut dengan mengidentifikasi item individual yang sering muncul dalam basis data dan memperluasnya ke set item yang lebih besar selama set tersebut muncul cukup sering dalam basis data. Itemset yang sering muncul yang ditentukan oleh Apriori dapat digunakan untuk menentukan aturan asosiasi yang menyoroti tren umum dalam basis data: ini memiliki aplikasi dalam domain seperti analisis keranjang pasar.

## Business Understanding
Market Basket Analisys adalah salah satu teknik kunci yang digunakan oleh pengecer besar untuk mengungkapkan asosiasi antara barang. Ini bekerja dengan mencari kombinasi barang yang sering muncul bersama dalam transaksi. Dengan kata lain, ini memungkinkan pengecer untuk mengidentifikasi hubungan antara barang yang dibeli oleh orang.

Aturan Asosiasi secara luas digunakan untuk menganalisis data keranjang belanja atau transaksi ritel dan dimaksudkan untuk mengidentifikasi aturan yang kuat yang ditemukan dalam data transaksi menggunakan ukuran ketertarikan, berdasarkan konsep aturan yang kuat.

Bagian Laporan ini mencakup:

### Problem Statements

Penjual item yang masif pada toko toko besar terkadang tidak terkontrol stok nya

### Goals

- Memudahkan menentukan stok berdasarkan barang cepat laku dan tidak laku

- Memudahkan menentukan diskon dan bundle paket penjualan

### Solution Statements

- Aturan Asosiasi secara luas digunakan untuk menganalisis data keranjang belanja atau transaksi ritel dan dimaksudkan untuk mengidentifikasi aturan yang kuat yang ditemukan dalam data transaksi menggunakan ukuran ketertarikan, berdasarkan konsep aturan yang kuat

- Apriori adalah algoritma untuk menambang itemset yang sering muncul dan pembelajaran aturan asosiasi pada basis data relasional
    

## Data Understanding

Dataset ini memiliki 38.765 baris dari pesanan pembelian orang-orang dari toko kelontong. Pesanan ini dapat dianalisis, dan aturan asosiasi dapat dihasilkan menggunakan Analisis Keranjang Pasar dengan algoritma seperti Algoritma Apriori.

[Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset/data)

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Member_number : Unique ID pada member dengan tipe data int
- Date : Tanggal transaksi
- itemDescription : Deskripsi Item

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
