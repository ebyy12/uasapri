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
Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan

``` bash 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
``` 

Selanjutnya kita mount dataset nya

``` bash 
df=pd.read_csv('/content/groceries-dataset/Groceries_dataset.csv')
``` 
Lalu tipe data dari masing-masing kolom, kita bisa menggunakan properti info,

``` bash 
df.info()
```

selanjutnya kita coba visualkan datanya agar mudah tebaca
```bash
top_10_item = df['itemDescription'].value_counts().nlargest(10)

plt.figure(figsize=(8, 8))
plt.pie(top_10_item, labels=top_10_item.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Produk Terlaris')
plt.show()
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/4ca1f750-6548-4024-8cb8-c7c79de17d5d)

atau kita bisa lihat jumlah eksaknya
```bash
plt.figure(figsize = (15,5))
bars = plt.bar(x = np.arange(len(freq.head(10))), height = (freq).head(10))
plt.bar_label(bars, fontsize=12, color='cyan', label_type = 'center')
plt.xticks(ticks = np.arange(len(freq.head(10))), labels = freq.index[:10])

plt.title('Top 10 Products by Support')
plt.ylabel('Support')
plt.xlabel('Product Name')
plt.show()
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/cd027210-3c21-4dde-949e-846b91cfcac6)

Kita bisa lihat frekwensi member dalam melakukan pembelian
```bash
member_shopping_frequency = df.groupby('Member_number')['Date'].count().sort_values(ascending=False)
sns.distplot(member_shopping_frequency, bins=8, kde=False, color='skyblue')
plt.xlabel('Number of purchasing')
plt.ylabel('Number of Member')
plt.title('member_shopping_frequency')
plt.show()
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/415c92c0-0978-4539-86e4-f8b72404b3e5)
 Kita juga bisa mencarinya top pembelian
 ```bash
plt.figure(figsize=(25,10))
sns.lineplot(x = df.itemDescription.value_counts().head(25).index, y = df.itemDescription.value_counts().head(25).values)
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top items purchased by members', color = 'green', size = 15)
plt.show()
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/747ffab5-791d-49c5-a7a6-a991e79b3e87)


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
