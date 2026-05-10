Tentu! Masalah utama yang kita hadapi pada *run* sebelumnya adalah **overfitting yang sangat parah**, di mana model Anda mampu menghafal data *training* dengan sangat baik (Train F1 mencapai ~0.97), tetapi gagal menggeneralisasi pada data validasi (Val F1 tertahan di ~0.66). 

Untuk mengatasi hal ini, saya telah menerapkan 5 perubahan strategis pada kode Anda. Berikut adalah penjelasan mengenai apa yang diubah beserta efeknya:

### 1. *Partial Backbone Freezing* (Pembekuan Sebagian Backbone)
*   **Perubahan:** Sebelumnya, pada epoch ke-8, Anda membuka kunci (*unfreeze*) seluruh parameter ResNet-50 dan BERT. Sekarang, saya memodifikasi fungsi `set_backbone_grad` agar **hanya membuka layer paling atas** saja (ResNet hanya `layer4`, dan BERT hanya layer `10`, `11`, dan `pooler`). Layer-layer di bawahnya akan tetap terkunci (*frozen*) secara permanen.
*   **Efek:** Dataset Anda relatif kecil (~4.500 sampel). Melakukan *fine-tuning* pada jutaan parameter secara keseluruhan membuat model langsung "menghafal" data training. Dengan hanya melatih layer atas, model dipaksa untuk menggunakan pengetahuan bahasa/visual asli dari *pre-trained weights*, dan hanya mengadaptasikan fitur tingkat tinggi (*high-level features*) untuk sentimen Anda.

### 2. Penambahan *Label Smoothing* pada Target EDL
*   **Perubahan:** Mengubah target `y_onehot` yang kaku (contoh: `[1.0, 0.0, 0.0]`) menjadi target yang lebih "lembut" (*soft targets*) menggunakan `epsilon = 0.1` (menjadi `[0.9, 0.05, 0.05]`).
*   **Efek:** Evidential Deep Learning (EDL) rentan menjadi *over-confident* (terlalu percaya diri) pada data training yang memicu jumlah *evidence* meroket tak terhingga. *Label smoothing* mencegah model merasa 100% benar pada data latih, memaksanya untuk menyisakan ruang ketidakpastian (*uncertainty*) sehingga jauh lebih tahan banting pada data validasi.

### 3. Penyesuaian Hyperparameter (*Regularization* Ekstra)
*   **`WEIGHT_DECAY` (0.02 → 0.05):** Hukuman L2 untuk bobot model dinaikkan agar nilai bobot tetap kecil dan mencegah model menjadi terlalu kompleks.
*   **`BATCH_SIZE` (32 → 16):** Ukuran *batch* yang lebih kecil memasukkan lebih banyak *noise* ke dalam perhitungan gradien. Hal ini bertindak sebagai penyeimbang alami (*natural regularizer*) agar model tidak mudah terjebak dalam penghafalan.
*   **`BACKBONE_LR` (5e-6 → 1e-6):** *Learning rate* untuk *backbone* diperkecil lagi agar bobot pada layer yang dibuka (*unfrozen*) tidak berubah terlalu drastis dan tidak merusak fitur bawaan *pre-trained*.

### 4. Peningkatan Penalti R-Drop
*   **Perubahan:** Pada fungsi `total_loss`, saya menaikkan pengali (*multiplier*) loss R-Drop dari `0.5` menjadi `1.5`.
*   **Efek:** R-Drop memaksa model untuk memprediksi hasil yang konsisten dari dua proses *forward-pass* dengan *dropout mask* yang berbeda pada sampel yang sama. Menekan model agar lebih konsisten (pengali 1.5) akan mencegahnya bergantung pada subset neuron tertentu untuk menghafal data training.

### 5. Augmentasi Teks (*Easy Data Augmentation* / EDA)
*   **Perubahan:** Sebelumnya Anda sudah menggunakan *Mixup* untuk gambar, tapi teks tidak diaugmentasi. Saya menambahkan fungsi Python sederhana `random_swap` ke dalam kelas dataset. Fungsi ini menukar posisi dua kata secara acak di dalam teks dengan probabilitas 15% pada saat training.
*   **Efek:** Meskipun perubahannya kecil, BERT kini akan melihat struktur kalimat yang sedikit berbeda pada setiap epoch. Ini mencegah *text-backbone* mengingat frasa atau kata spesifik dari training set, dan mendorong model untuk lebih memahami sinyal sentimen secara makro.

### 6. Relaksasi EDL Regularizer
*   **Perubahan:** `NEUTRAL_BOOST` diturunkan dari 2.0 menjadi `1.0`, dan `KL_ANNEALING_EPOCHS` dinaikkan dari 5 menjadi `10`.
*   **Efek:** Memberikan waktu tambahan bagi model untuk belajar memetakan *evidence* secara alami selama 10 epoch awal sebelum penalti *KL-divergence* ditekan kuat. Penurunan *neutral boost* dilakukan agar model tidak bereaksi berlebihan dan "menebak" terlalu banyak Neutral (mencegah tingginya *false positives* pada Neutral).

**Secara Keseluruhan:**
Setelah Anda me-*run* kembali notebook ini, **skor Train F1 kemungkinan akan lebih sulit naik ke 0.97** (mungkin tertahan di 0.85 - 0.90 karena dipaksa lebih "bodoh" saat training), namun sebaliknya, model tidak akan terjebak overfitting dan **Val F1 memiliki potensi yang jauh lebih besar untuk naik menembus 0.70 - 0.80.** 
