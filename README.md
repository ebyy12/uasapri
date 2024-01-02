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

kita juga bisa melihat persentase pembelian perbulan nya stelah kita mensplit format tanggal nya
```bash
df['Date'] = pd.to_datetime(df['Date'])
```
```bash
df['date'] = df['Date'].dt.date

df['month'] = df['Date'].dt.month
df['month'] = df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12),
                                          ('January','February','March','April','May','June','July','August',
                                          'September','October','November','December'))

df['weekday'] = df['Date'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6),
                                          ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))

df.drop('Date', axis = 1, inplace = True)
```
Jika sudah di split tinggal masukan
```bash
months = df.groupby('month')['Member_number'].count()
pie, ax = plt.subplots(figsize=[10,6])
plt.pie(x=months, autopct="%.1f%%", explode=[0.05]*12, labels=months.keys(), pctdistance=0.5)
plt.title("Items bought split by month", fontsize=14)
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/02ba3ebe-745a-4e76-ae94-03b379bae67b)

Lalu kita bisa lihat penjualan harianya 
```bash
plt.figure(figsize=(12,5))
sns.barplot(x=days.keys(), y=days)
plt.xlabel('Week Day', size = 15)
plt.ylabel('Orders per day', size = 15)
plt.title('Number of orders received each day', color = 'blue', size = 15)
plt.show()
```
![image](https://github.com/ebyy12/uasapri/assets/148988993/f4dceee9-a10d-45c6-a111-1d09ca790df8)




## Modeling
Selanjutnya kita lanjutkan modeling 
```bash
df['itemDescription'] = df.groupby(['Member_number', 'date'])['itemDescription'].transform(lambda x: ','.join(x))
```
lalu kita bagi item yang disatukan oleh coma
```bash
lst=[]
for i in range(0,len(df)-1):
    data = df['itemDescription'][i].split(',')
    lst.append(data)
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
