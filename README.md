# Laporan Proyek Machine Learning - Ibnu Raju Humam

[LinkedIn - Ibnu Raju Humam](https://www.linkedin.com/in/ibnu-raju-humam-a14502221/)

## Project Overview

Pada era digital sekarang ini, masyarakat semakin dimudahkan untuk mengakses dan membaca buku melalui platform digital [[1]][[2]]. Berbagai situs dan aplikasi bahkan memberikan kesempatan kepada pengguna untuk membaca e-book, memberikan ulasan, dan menilai buku yang mereka baca seperti pada [Google Play Book], [Goodreads], dan [Amazon Book]. Namun, dengan melimpahnya pilihan buku dari berbagai penulis, genre, bahkan penerbit, pengguna terkadang mengalami kesulitan dalam memilih buku yang sesuai dengan minat dan kebutuhannya. Oleh karenanya, perlu dikembangkan model yang mampu membantu menyaring informasi dan menyarankan buku yang relevan dengan pembaca.

Sistem rekomendasi menjadi solusi yang penting dalam permasalahan ini. Dengan memanfaatkan data interaksi pengguna dan karakteristik buku, sistem rekomendasi mampu menyarankan buku yang sesuai dengan preferensi pengguna. Pendekatan seperti content-based filtering, yang merekomendasikan item serupa berdasarkan fitur konten, serta collaborative filtering, yang memanfaatkan kesamaan perilaku antar pengguna, telah terbukti efektif dalam meningkatkan kepuasan dan keterlibatan pengguna [[3]][[4]]. Sistem ini tidak hanya menghemat waktu, tetapi juga meningkatkan kemungkinan pengguna menemukan buku yang sesuai dengan selera mereka masing-masing.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku dengan memanfaatkan dua pendekatan yaitu content-based filtering dan collaborative filtering. Sistem ini dirancang menggunakan [Book Recommendation Dataset], yang merupakan dataset publik dan tersedia di platform Kaggle. Dataset ini mencakup data tentang buku, pengguna, dan penilaian yang diberikan. Dengan mengembangkan model rekomendasi berbasis data ini, proyek ini diharapkan mampu memberikan pengalaman membaca yang lebih personal dan efisien bagi pengguna.

## Business Understanding
### Problem Statements
1. Pernyataan Masalah Ke-1
Pengguna kesulitan menemukan buku yang sesuai dengan preferensi mereka. Terkadang pengguna membaca menyesuaikan dengan genre bacaanya, misal menyukai buku fantasi, atau buku sejarah. Selain itu, terkadang juga terdapat pembaca yang mempertimbangkan berdasarkan siapa penulisnya, diterbitkan oleh publisher mana, atau preferensi-preferensi lainnya.
2. Pernyataan Masalah Ke-2
Banyaknya pilihan buku membuat pengalaman pencarian menjadi kurang efisien. Dengan banyaknya pilihan ini, bisa memakan waktu yang begitu lama. Apalagi pertimbangan preferensi individu yang bisa saja bukan sekedar berdasarkan genre dari buku tersebut, melainkan juga misal penulisnya, penerbit, tahun terbit dan lain-lain. Oleh karena itu, dibutuhkan sistem yang mampu mempersonalisasi rekomendasi agar pengalaman pengguna dalam mencari dan menemukan buku menjadi lebih cepat, relevan, dan menyenangkan.

### Goals
1.	Menjawab Pernyataan Masalah Ke-1
Mengembangkan sistem rekomendasi buku yang dapat memberikan saran secara personal. Dengan memeberikan saran buku yang dipersonalisasikan, pengguna akan dengan mudah menemukan dan memutuskan untuk membaca buku yang mana, baik berdasarkan genre, penulis, maupun penerbit. Hal ini akan membantu mempercepat proses pengambilan keputusan dalam memilih buku untuk dibaca.
2.	Menjawab Pernyataan Masalah Ke-2
Menyediakan dua pendekatan sistem rekomendasi yaitu content-based dan collaborative filtering. Dua pendekatan sistem ini telah terbukti dapat memberikan penawaran buku yang sesuai dengan preferensi pengguna. Dengan adanya sistem ini, pengguna tidak perlu lagi menelusuri seluruh koleksi secara manual untuk menemukan buku yang sesuai dengan minat mereka.

### Solution Statements
1.	Solusi 1
Menggunakan pendekatan content-based filtering untuk merekomendasikan buku berdasarkan kemiripan fitur buku.
2.	Solusi 2
Menerapkan collaborative filtering untuk memberikan rekomendasi berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa. Dengan pendekatan ini memungkinkan sistem mengetahui hubungan antara pengguna dan buku berdasarkan nilai rating yang diberikan.
3.	Solusi 3
Melakukan evaluasi terhadap masing-masing pendekatan dengan metrik yang sesuai:
    - Untuk content-based filtering: menggunakan **Precision@K** berbasis _rating_ sebagai indikator relevansi rekomendasi (karena tidak ada label genre).
    - Untuk collaborative filtering: menggunakan **Root Mean Squared Error (RMSE)** untuk mengukur seberapa akurat prediksi _rating_ yang diberikan oleh sistem terhadap data sebenarnya.

## Data Understanding
_Dataset_ yang digunakan dalam proyek ini adalah [Book Recommendation Dataset] yang tersedia secara publik di Kaggle. _Dataset_ ini relevan digunakan untuk mengembangkan model rekomendasi prediktif dengan ML karena menyediakan beberapa indikator penting seperti ISBN, judul buku, penulis serta penerbit dan lain-lain. _Dataset_ ini terdiri dari tiga bagian utama:

#### Books.csv
- ISBN: International Standard Book Number, kode unik identifikasi setiap buku
- Book-Title: Judul buku
- Book-Author: Nama penulis buku
- Year-Of-Publication: Tahun terbit buku
- Publisher: Nama penerbit buku
- Image-URL-S: URL untuk gambar sampul buku berukuran kecil (small)
- Image-URL-M: URL untuk gambar sampul buku berukuran sedang (medium)
- Image-URL-L: URL untuk gambar sampul buku berukuran besar (large)

#### Users.csv
- User-ID: Nomor unik identifikasi pengguna
- Location: Lokasi pengguna, dengan format "kota, negara bagian (opsional), negara"
- Age: Usia pengguna

#### Ratings.csv
- User-ID: Nomor identifikasi pengguna (mengacu pada Users.csv)
- ISBN: Nomor identifikasi buku (mengacu pada Books.csv)
- Book-Rating: Skor penilaian dari pengguna terhadap buku, dengan skala 0–10. Nilai 0 menunjukkan bahwa pengguna tidak memberikan penilaian eksplisit terhadap buku tersebut (implicit feedback).

### Exploratory Data Analysis (EDA)
#### books
**Books.csv** berisi informasi metadata mengenai buku, dengan total 271.360 baris data. Dataset ini mencakup berbagai atribut penting yang menjelaskan identitas setiap buku. Adapun fitur-fitur yang terdapat dalam dataset ini meliputi ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L.
```sh
books.info()
```
| No | Column               | Non-Null Count | Dtype  |
|----|----------------------|----------------|--------|
| 0  | ISBN                 | 271360         | object |
| 1  | Book-Title           | 271360         | object |
| 2  | Book-Author          | 271358         | object |
| 3  | Year-Of-Publication  | 271360         | object |
| 4  | Publisher            | 271358         | object |
| 5  | Image-URL-S          | 271360         | object |
| 6  | Image-URL-M          | 271360         | object |
| 7  | Image-URL-L          | 271357         | object |

Jumlah unik ISBN dalam `books` sebanyak 271.360, jumlah unik penulis sebanyak 102.022 dan jumlah unik penerbit sebanyak 16.807. Dari hasil ini bisa diliat bahwa beberapa penulis memiliki lebih dari satu buku dengan ISBN yang berbeda. Kemudian, beberapa penulis bisa jadi menerbitkan bukunya di penerbit yang sama.

```sh
print('Jumlah ISBN unik berdasarkan books: ', books['ISBN'].nunique())
print('Jumlah penulis: ', books['Book-Author'].nunique())
print('Jumlah penerbit: ', books['Publisher'].nunique())

output:
# Jumlah ISBN unik berdasarkan books:  271360
# Jumlah penulis:  102022
# Jumlah penerbit:  16807
```
Pada bagian ini untuk mengetahui 10 penulis dengan buku terbanyak. Berdasarkan gambar, Agatha Christie menempati posisi teratas dengan buku terbanyak, diikuti Willian Shakespeare, Stephen King, dan seterusnya.
![Top 10 Author](https://github.com/26humam/book_recommendation/blob/main/top_10_author.png?raw=true)

Pada bagian ini untuk mengetahui 10 penerbit dengan terbitan buku terbanyak. Harlequin Publisher menempati posisi pertama, diikutii Silhouette, Pocket dan seterusnya seperti di gambar.
![Top 10 Publisher](https://github.com/26humam/book_recommendation/blob/main/top_10_publisher.png?raw=true)

#### users
**Users.csv** berisi informasi mengenai pengguna, dengan total 278.858 baris data. Dataset ini mencakup identitas pengguna berupa User-ID, Location dan Age.
```sh
users.info()
```
| No | Column   | Non-Null Count | Dtype    |
|----|----------|----------------|----------|
| 0  | User-ID  | 278858         | int64    |
| 1  | Location | 278858         | object   |
| 2  | Age      | 168096         | float64  |

Berdasarkan data Users.csv, banyaknya pengguna unik sebanyak 278.858 dan lokasi pengguna unik sebanyak 57.339 yang tersebar diseluruh dunia.
```sh
print('Jumlah pengguna unik berdasarkan users: ', users['User-ID'].nunique())
print('Jumlah lokasi pengguna: ', len(users.Location.unique()))

output:
# Jumlah pengguna unik berdasarkan users:  278858
# Jumlah lokasi pengguna:  57339
```

Pengguna terbanyak berasal dari Negara USA, diikuti Kanada, United Kingdom, Germany dan Spain. 
![Top 10 Aaal Pengguna](https://github.com/26humam/book_recommendation/blob/main/top_10_user_country.png?raw=true)

#### ratings
**Ratings.csv** berisi informasi mengenai rating yang diberikan pengguna terhadap buku tertentu. Dalam data ini terdiri dari kolom User-ID, ISBN, dan Book-Rating dengan tipe data yang dapat dilihat pada tabel di bawah.
```sh
ratings.info()
```
| No | Column       | Non-Null Count  | Dtype    |
|----|--------------|-----------------|----------|
| 0  | User-ID      | 1149780         | int64    |
| 1  | ISBN         | 1149780         | object   |
| 2  | Book-Rating  | 1149780         | int64    |

Untuk jumlah user unik dalam pemberian rating oleh pengguna ini ada sebanyak 105.283 rating yang diberikan dan buku yang dinilai berdasarkan ISBN sebanyak 340.556 buku.
```sh
print('Jumlah User-ID unik berdasarkan ratings: ', ratings['User-ID'].nunique())
print('Jumlah ISBN unik berdasarkan ratings: ', ratings['ISBN'].nunique())

output:
# Jumlah User-ID unik berdasarkan ratings:  105283
# Jumlah ISBN unik berdasarkan ratings:  340556
```

## Data Preparation
Tahapan data preparation atau data preprocessing dilakukan untuk memastikan data yang akan digunakan selama proses pelatihan model prediktif berada dalam kondisi optimal, tidak ada _noise_, konsisten, serta memenuhi keperluan yang dibutuhkan oleh model. Adapun yang dilakukan pada proyek ini adalah sebagai berikut:

1.	Menggabungkan Buku dan Ratings
    Langkah pertama adalah menggabungkan data daru dua sumber yaitu Books.cvs dan Ratings.csv. Hal yang perlu diperhatikan terlebih dahulu, bahwa dataset pada proyek ini sangat besar, sehingga diputuskan yang akan digunakan dalam proyek ini hanya 100 buku (berdasarkan ISBN) yang paling banyak diberi rating oleh pengguna atau 100 buku terpopuler.
    ```sh
    top_books = ratings['ISBN'].value_counts().head(100).index
    ```
    
    Perlu dipastikan juga ISBN pada `top_books` terdapat juga pada data `books`. Kemudian, diasumsikan rating dengan nilai 0 berarti tidak memberikan penilaian terhadap buku. Dan, karena satu buku bisa saja memiliki banyak penilaian dari pengguna, akan diambil nilai rata-rata dari rating per buku.
    ```sh
    ratings_filtered = ratings[
    (ratings['ISBN'].isin(books['ISBN'])) &
    (ratings['Book-Rating'] > 0)
    ]
    
    mean_rating_per_book = (
            ratings_filtered
            .groupby('ISBN')['Book-Rating']
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={'Book-Rating': 'Average_Rating'})
    )
    ```
    Data gabungan akan disimpan pada variabel `df_top`
    ```sh
    df_isbn = mean_rating_per_book.merge(books, on='ISBN', how='inner')
    df_top = df_isbn[df_isbn['ISBN'].isin(top_books)].reset_index(drop=True)
    ```
    
    Namun, ada satu data hasil rating dengan ISBN yang tidak terdaftar pada ISBN berdasarkan data `books`, sehingga akan otomatis tidak digabungkan pada df_top. Output-nya hanya menunjukkan 99 buku.
    ```sh
    df_top.shape
    
    output:
    # (99, 9)
    ```

2.	Drop Kolom
   Kolom `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L` pada `df_top` dihapus karena tidak memberikan kontribusi terhadap pembentukan sistem rekomendasi. Ketiga kolom ini hanya berisi tautan gambar sampul buku dalam berbagai ukuran, dalam bentuk gambar dan tidak memiliki nilai informatif. Data final akan disimpan pada `df_final`.
    ```sh
    df_final = df_top.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=False)
    ```

3.	Penanganan Mission Values & Duplicate
    Pada data `df_final` sudah tidak ditemukan lagi data dengan _missing values_ dan data _duplicate_, karena pada proses sebelumnya sudah dipastikan data unik untuk setiap bukunya berdasarkan ISBN.

## Modeling
Tahapan modeling merupakan inti dari proses sistem rekomendasi yang dibangun dalam proyek ini. Pada tahap ini, sistem dilatih dan diuji untuk memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan data yang tersedia. Terdapat dua pendekatan utama yang digunakan, yaitu:
1.	Content-Based Filtering
2.	Collaborative Filtering

### 1.	Content-Based Filtering
Content-Based Filtering adalah metode sistem rekomendasi yang menyarankan item kepada pengguna berdasarkan kemiripan atribut atau fitur konten lainnya, dalam proyek ini menyarankan buku. Dalam proyek ini, fitur-fitur seperti `Book-Title`, `Book-Author`, dan `Publisher` digunakan untuk menghitung tingkat kemiripan antar buku dengan menggunakan metode **TF-IDF (Term Frequency–Inverse Document Frequency)** dan **cosine similarity**. Pendekatan ini membantu memberikan rekomendasi meskipun tidak tersedia data eksplisit seperti genre. Kemudian, memberikan rekomendasi buku dengan skor kemiripan tertinggi terhadap judul buku.

#### TF-IDF Vectorizer
```sh
data['combined-features'] = (
    data['Book-Title'].fillna('') + " " +
    data['Book-Author'].fillna('') + " " +
    data['Publisher'].fillna('')
)
```
Menggabungkan fitur-fitur teks menjadi satu kolom (combined-features) agar bisa dianalisis sebagai satu dokumen teks per buku.

```
tf = TfidfVectorizer(stop_words='english')
tf.fit(data['combined-features'])
tfidf_matrix = tf.fit_transform(data['combined-features'])
```
Mengubah teks menjadi vektor numerik menggunakan TF-IDF, yang memberi bobot tinggi pada kata-kata yang unik di suatu buku namun jarang muncul di buku lain.

#### Cosine Similarity
```
cosine_sim = cosine_similarity(tfidf_matrix)
```
Menghitung tingkat kemiripan antar buku berdasarkan vektor TF-IDF menggunakan **cosine similarity** (semakin tinggi nilainya, semakin mirip).

#### Fungsi Rekomendasi dan Fungsi Presisi Berdasarkan Rating
Fungsi `book_recommendations()` akan mengambil input judul buku, menghitung kemiripan dengan buku lain, lalu mengembalikan daftar buku yang paling mirip berdasarkan skor similarity tertinggi. Sedangkan, fungsi `precision_rating_based()` digunakan untuk mengevaluasi apakah buku yang direkomendasikan memiliki rating yang tinggi atau tidak (sebagai relevansi). Fungsi `precision_rating_based()` akan dijelaskan pada bagian **Evaluation**

### 2.	Collaborative Filtering
Collaborative Filtering adalah metode yang memberikan rekomendasi dengan menganalisis preferensi pengguna lain yang memiliki pola perilaku serupa. Pendekatan ini mengandalkan data interaksi (rating) antar pengguna dan item tanpa memperhatikan fitur konten dari buku. Data yang digunakan adalah data `ratings_filtered` yang disimpan pada `df`, seperti pada penjelasan Bagian **Data Preparation**. User-ID dan ISBN di-encode menjadi angka numerik. Model belajar embedding untuk setiap pengguna dan ISBN, kemudian menggunakan produk dot dari embedding ini untuk memprediksi.
```sh
user_ids = df['User-ID'].unique().tolist()
print('5 contoh User-ID: ', user_ids[:5])

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('5 contoh encoded User-ID : ', list(user_to_user_encoded.items())[:5])

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('5 contoh encoded angka ke User-ID: ', list(user_encoded_to_user.items())[:5])
```

Sehingga hasil akhir user yang didapatkan sebanyak 68.091 dan jumlah ISBN sebanyak 149.836 buku.

#### Membagi Data Training dan Data Validasi
Selanjutnya membagi data untuk Training dan Validasi. Sebelumnya data dibagi menjadi variabel `x` dan variabel `y`, kemudian dibagi menjadi 80% untuk training dan 20% untuk validasi.
```sh
# Membuat variabel x untuk mencocokkan data user dan isbn menjadi satu value
x = df[['user', 'isbn']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

#### Proses Training
Membuat model rekomendasi berbasis Neural Collaborative Filtering dengan `tf.keras.Model`. Kemudian model diinisialisasi dengan jumlah pengguna dan buku, serta ukuran embedding sebesar 50.
```sh
model = RecommenderNet(num_users, num_isbn, 50)
```
Selanjutnya didefinisikan juga _compile_ menggunakan `BinaryCrossentropy` karena model ini mengoutput nilai antara 0–1 (relevan/tidak relevan), menggunakan optimizer `Adam` dan **RootMeanSquaredError (RMSE)** dipakai untuk evaluasi seberapa jauh prediksi dari label asli. Selanjutnya model ditraining dengan `epoch` 20.
```sh
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 20,
    validation_data = (x_val, y_val)
)
```

#### Fungsi Rekomendasi
Bagian rekomendasi buku menggunakan pendekatan Collaborative Filtering ini bertujuan untuk menyarankan buku yang kemungkinan besar disukai oleh pengguna, berdasarkan pola interaksi pengguna lain yang memiliki preferensi serupa. Dengan memilih satu pengguna secara acak, sistem mengidentifikasi buku-buku yang belum pernah dibaca oleh pengguna tersebut, kemudian memprediksi skor ketertarikan (rating) untuk setiap buku menggunakan model neural network yang telah dilatih. Dari hasil prediksi ini, sistem menampilkan 10 buku teratas dengan skor tertinggi sebagai rekomendasi

## Evaluation
### Fungsi Presisi Berdasarkan Rating
Fungsi `precision_rating_based()` digunakan untuk mengevaluasi seberapa relevan rekomendasi buku yang dihasilkan oleh sistem Content-Based Filtering, berdasarkan rating pengguna yang sudah ada.
```sh
def precision_rating_based(recommended_books, rating_column='Average-Rating', min_rating=7.0):
    if recommended_books.empty:
        return 0.0
    total = len(recommended_books)
    relevan = recommended_books[rating_column] >= min_rating
    return relevan.sum() / total
```
Karena pada dataset tidak tersedia data genre, maka rating digunakan sebagai pertimbangan kualitas atau relevansi buku. Presisi dipilih karena cocok untuk mengevaluasi seberapa banyak hasil rekomendasi yang benar-benar bernilai bagi pengguna, dengan kondisi tidak tahu preferensi personal mereka secara langsung. Nilai ambang batas pada kasus ini disetel pada 7,0.

### Metrik Evaluasi: RMSE (Root Mean Squared Error)
Untuk mengevaluasi kinerja model Collaborative Filtering, digunakan metrik **Root Mean Squared Error (RMSE)**. MSE mengukur seberapa jauh nilai prediksi model dari nilai aktual. Formula RMSE adalah akar kuadrat dari rata-rata kuadrat perbedaan antara nilai prediksi dan nilai aktual. Semakin rendah nilai RMSE, semakin baik kinerja model dalam memprediksi rating.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Keterangan:
- n = jumlah sampel
- $$y_i$$ = nilai aktual (true value)
- $$\hat{y}_i$$ = nilai prediksi


### Hasil Evaluasi & Analisis
#### 1. Content-Based Filtering
Dalam pendekatan Content-Based Filtering, sistem rekomendasi memberikan saran buku yang memiliki kemiripan dalam atribut konten, yaitu judul buku, nama penulis, dan penerbit. Pada studi kasus ini, digunakan buku **Bel Canto: A Novel** sebagai input untuk menghasilkan rekomendasi.

| ISBN        | Average-Rating | Book-Title         | Book-Author   | Year-of-Publication | Publisher  | Combined-Features|
|-------------|----------------|--------------------|---------------|---------------------|------------|------------------|
| 0060934417  | 8.2            | Bel Canto: A Novel | Ann Patchett  | 2002                | Perennial  | Bel Canto: A Novel Ann Patchett Perennial |

```sh
book_recommendations('Bel Canto: A Novel')
```
| No | Book-Title | Book-Author | Publisher | Average-Rating | Score-Similarity |
|----|------------|---------|----------|------------------|----------------|
| 1 | Little Altars Everywhere: A Novel | Rebecca Wells | Perennial | 7.6 | 0.211 |
| 2 | The Poisonwood Bible: A Novel | Barbara Kingsolver | Perennial | 8.2 | 0.190 |
| 3 | Divine Secrets of the Ya-Ya Sisterhood: A Novel | Rebecca Wells | Perennial | 7.9 | 0.148 |
| 4 | Fall On Your Knees (Oprah #45) | Ann-Marie MacDonald | Touchstone | 7.6 | 0.143 |
| 5 | Fast Food Nation: The Dark Side of the All-American Meal | Eric Schlosser | Perennial | 8.4 | 0.096 |

Untuk mengukur seberapa relevan hasil rekomendasi, digunakan metrik precision berbasis rating, yaitu proporsi rekomendasi yang memiliki rating di atas ambang batas tertentu, dalam contoh ini yaitu 7.0 (lihat pada bagian **Evaluation**)
```sh
rekomendasi = book_recommendations("Bel Canto: A Novel", k=5)
precision = precision_rating_based(rekomendasi)
print(f"Precision berbasis rating: {precision:.2f}")

Output:
# Precision berbasis rating: 1.00
```
Semua buku yang direkomendasikan memiliki rata-rata rating di atas 7.0 (ambang batas), yang berarti 100% dari rekomendasi dianggap relevan menurut metrik evaluasi ini. Ini menunjukkan bahwa sistem berhasil menyarankan buku-buku berkualitas tinggi berdasarkan preferensi kontennya.

#### 2. Collaborative Filtering

![Matriks Evaluasi](https://github.com/26humam/book_recommendation/blob/main/matriks_evaluasi.png?raw=true)
- Train RMSE menurun secara konsisten hingga ~0.1349
- Test RMSE stabil di sekitar ~0.1948
- Tidak terjadi overfitting yang signifikan, model menunjukkan generalisasi yang baik. Selisih Train RMSE dan Test RMSE hanya ~0,06

```sh
recommended_book = isbn_df[isbn_df['ISBN'].isin(recommended_book_ids)]
for idx, row in recommended_book.iterrows():
    print(row['Book-Title'], ':', row['Book-Author'])

Output:
# Showing recommendations for users: 28360
# ===========================
# Book with high ratings from user
# --------------------------------
# Gon : Introducing The Dinosaur That Time Will Never Forget! (Paradox Fiction) : Masashi Tanaka
# Teacher's Pet : Richie Tankersley Cusick
# The Stinky Cheese Man and Other Fairly Stupid Tales : Jon Schieszka
# Only Child (An Avon Flare Book) : Jesse Osburn
# Myst: The Book of Atrus : Robyn Miller
# --------------------------------
# Top 10 book recommendation
# --------------------------------
# Where the Sidewalk Ends : Poems and Drawings : Shel Silverstein
# Postmarked Yesteryear: 30 Rare Holiday Postcards : Pamela E. Apkarian-Russell
# Fox in Socks (I Can Read It All by Myself Beginner Books) : Dr. Seuss
# The Two Towers (The Lord of the Rings, Part 2) : J. R. R. Tolkien
# The Giving Tree : Shel Silverstein
# The Lorax : Dr. Seuss
# Dilbert: A Book of Postcards : Scott Adams
# The Sneetches and Other Stories : Dr. Seuss
# Harry Potter and the Chamber of Secrets Postcard Book : J. K. Rowling
# The Hitchhiker's Guide to the Galaxy : Douglas Adams
```

Model collaborative filtering berhasil memberikan rekomendasi buku yang relevan dengan preferensi pengguna, ditunjukkan oleh performa error yang rendah dan daftar buku yang berkaitan dengan minat pengguna sebelumnya.

## Kesimpulan Akhir
Kedua pendekatan menunjukkan kinerja yang baik dan saling melengkapi.
- Meskipun tidak ada informasi eksplisit seperti genre, sistem tetap dapat memberikan rekomendasi relevan.
- Evaluasi menggunakan precision berbasis rating menunjukkan nilai yang baik, artinya semua buku yang direkomendasikan berkualitas baik (rating di atas rata-rata).
- Model berbasis neural network menunjukkan kinerja baik dengan Train RMSE ~0.13 dan Test RMSE ~0.19, menandakan model generalisasi dengan baik.
- Buku-buku yang direkomendasikan sesuai dengan pola minat pengguna, sehingga mempermudah proses pengambilan keputusan.

Sistem rekomendasi yang dikembangkan berhasil memberikan rekomendasi pencarian buku yang lebih personal dan efisien, menjawab permasalahan awal terkait banyaknya pilihan dan kesulitan menentukan buku yang sesuai preferensi.

## Saran
- Menambahkan fitur Genre atau Sinopsis pada Content-Based Filtering. Saat ini, sistem rekomendasi Content-Based Filtering hanya mengandalkan fitur judul, penulis, dan penerbit. Penambahan fitur genre atau sinopsis akan memperkaya informasi konten buku, sehingga sistem dapat memberikan rekomendasi yang lebih mendalam dan akurat. Jika data lengkap tidak tersedia, fitur ini bisa ditambahkan secara manual pada 100 buku terpilih sebagai percobaan awal.
- Menerapkan Hybrid Filtering (Gabungan Content-Based dan Collaborative Filtering).
Menggabungkan kedua pendekatan akan membantu mengatasi keterbatasan masing-masing. 

## References
[[1]] “Mengapa Buku Digital Semakin Populer? Meninjau Fenomena Literasi di Era Digital,” cabjari-pangkalankotobaru , 20 Februari 2024. https://cabjari-pangkalankotobaru.kejaksaan.go.id/mengapa-buku-digital-semakin-populer-meninjau-fenomena-literasi-di-era-digital/ (diakses 31 Mei 2025).

[[2]] I Made Putrayasa, I Gede Suwindia, and I. Made, “Transformasi literasi di era digital: tantangan dan peluang untuk generasi muda,” Education and Social Sciences Review, vol. 5, no. 2, pp. 156–165, 2024, doi: https://doi.org/10.29210/07essr501400.

[[3]] S. Sharma, V. Rana, and M. Malhotra, “Automatic recommendation system based on hybrid filtering algorithm,” Education and Information Technologies, Jul. 2021, doi: https://doi.org/10.1007/s10639-021-10643-8.

[[4]] Ashlesha Bachhav, Apeksha Ukirade, Nilesh Patil, Manish Saswadkar, and Prof. Nitin Shivale, “Book Recommendation System using Machine learning and Collaborative Filtering,” International Journal of Advanced Research in Science, Communication and Technology, pp. 279–283, Dec. 2022, doi: https://doi.org/10.48175/ijarsct-7687.

   [1]: <https://cabjari-pangkalankotobaru.kejaksaan.go.id/mengapa-buku-digital-semakin-populer-meninjau-fenomena-literasi-di-era-digital/>
   [2]: <https://jurnal.iicet.org/index.php/essr/article/view/5014>
   [3]: <https://link.springer.com/article/10.1007/s10639-021-10643-8>
   [4]: <https://ijarsct.co.in/Paper7687.pdf>

   [Book Recommendation Dataset]: <https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset>
   
   [Google Play Book]:
   <https://play.google.com/store/books?hl=id>
   
   [Goodreads]:
   <https://www.goodreads.com/>
   
   [Amazon Book]:
   <https://www.amazon.com/books-used-books-textbooks/b?ie=UTF8&node=283155>
   
