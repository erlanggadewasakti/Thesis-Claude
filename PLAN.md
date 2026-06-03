# Plan Implementasi: Integrasi Evidential Deep Learning dan Bi-directional Cross-Attention Berbasis Feature Decoupling pada Analisis Sentimen Multimodal

## 📌 Deskripsi Proyek

Proyek ini membangun arsitektur _State-of-the-Art_ (SOTA) untuk Analisis Sentimen Multimodal (Teks dan Gambar) menggunakan PyTorch. Fokus utama adalah menangani _noise_ dan inkongruensi (seperti sarkasme) dengan mengintegrasikan _Evidential Deep Learning_ (EDL) sebagai pengukur ketidakpastian (_uncertainty_) yang mengontrol aliran _Bi-directional Cross-Attention_.

**Dataset:** MVSA-Single (3 Kelas: Negative, Neutral, Positive).

---

## 🛠️ Fase 1: Setup Environment & Data Pipeline

**Tujuan:** Memuat dataset, melakukan prapemrosesan teks/gambar, dan membuat PyTorch `DataLoader`.

- **Step 1.1:** Terapkan skrip prapemrosesan yang ada untuk membersihkan `labelResultAllFinal.txt` dan memuat string teks dengan penanganan _encoding_.
- **Step 1.2:** Buat kelas kustom `MVSADataset(Dataset)`.
  - **Teks:** Gunakan `transformers.AutoTokenizer` (misal: `roberta-base`).
  - **Gambar:** Gunakan `torchvision.transforms` (Resize ke 224x224, Normalize) atau `transformers.AutoImageProcessor` jika menggunakan ViT.
- **Step 1.3:** Buat modul `DataLoader` untuk _train, validation_, dan _test split_ (rasio standar 80:10:10).

---

## 🧠 Fase 2: Unimodal Feature Extraction Module

**Tujuan:** Mengekstraksi fitur mentah dari kedua modalitas menggunakan model _pre-trained_.

- **Step 2.1 (Text Encoder):** Implementasikan model Transformer (misal: `roberta-base`). Ekstrak _hidden state_ terakhir.
  - _Input:_ `input_ids`, `attention_mask`.
  - _Output:_ Vektor fitur teks $F_T \in \mathbb{R}^{d}$.
- **Step 2.2 (Image Encoder):** Implementasikan arsitektur Visi (misal: `resnet50` atau `vit-base-patch16-224`).
  - _Input:_ Tensor gambar.
  - _Output:_ Vektor fitur gambar $F_I \in \mathbb{R}^{d}$.
  - _Catatan:_ Lakukan proyeksi linear jika dimensi $F_T$ dan $F_I$ tidak sama.

---

## ✂️ Fase 3: Feature Decoupling Module

**Tujuan:** Memisahkan fitur menjadi representasi 'sentimen' dan 'latar belakang'.

- **Step 3.1:** Buat lapisan _Decoupling_ menggunakan Multi-Layer Perceptron (MLP) terpisah untuk setiap modalitas.
- **Step 3.2:** Ekstrak Fitur Sentimen Murni ($sent$) untuk diteruskan ke tahap atensi.
  - $F_{T,sent} = MLP_{T,sent}(F_T)$
  - $F_{I,sent} = MLP_{I,sent}(F_I)$
- _(Opsional)_ Terapkan _Orthogonal Loss_ antara fitur sentimen dan latar belakang agar model benar-benar memisahkan _noise_.

---

## 🔄 Fase 4: Bi-directional Cross-Attention (BCA) Module

**Tujuan:** Melakukan fusi asimetris dua arah untuk menghindari bias pada salah satu modalitas.

- **Step 4.1 (Aliran Teks-ke-Gambar / $T \rightarrow V$):** Teks menyoroti Gambar.
  - $Q = F_{T,sent} W_Q$
  - $K = F_{I,sent} W_K$ , $V = F_{I,sent} W_V$
  - $A_{T \rightarrow V} = Softmax(\frac{Q K^T}{\sqrt{d_k}}) V$
- **Step 4.2 (Aliran Gambar-ke-Teks / $V \rightarrow T$):** Gambar menyoroti Teks.
  - $Q = F_{I,sent} W_Q$
  - $K = F_{T,sent} W_K$ , $V = F_{T,sent} W_V$
  - $A_{V \rightarrow T} = Softmax(\frac{Q K^T}{\sqrt{d_k}}) V$

---

## ⚖️ Fase 5: Evidential Deep Learning (EDL) & Adaptive Gating

**Tujuan:** Menghitung ketidakpastian epistemik dari masing-masing aliran atensi untuk menyaring konflik fitur.

- **Step 5.1 (Evidential Layer):** Lewatkan $A_{T \rightarrow V}$ dan $A_{V \rightarrow T}$ ke lapisan _Linear_ + fungsi _Softplus_ untuk memastikan nilai evidensi tak negatif.
  - $e_k = Softplus(Linear(A))$ untuk kelas $k$ (Neg, Neu, Pos).
- **Step 5.2 (Dirichlet Parameters):** Hitung parameter distribusi Dirichlet.
  - $\alpha_k = e_k + 1$
  - $S = \sum_{k=1}^{K} \alpha_k$
- **Step 5.3 (Uncertainty Quantification):** Hitung _Belief_ ($b$) dan _Uncertainty_ ($u$).
  - $b_k = \frac{\alpha_k - 1}{S}$
  - $u = \frac{K}{S}$ (di mana $K=3$)
- **Step 5.4 (Adaptive Gating):** Gunakan nilai _uncertainty_ sebagai _gate_ (pembobot) untuk fitur.
  - $F'_{T \rightarrow V} = A_{T \rightarrow V} \cdot (1 - u_{T \rightarrow V})$
  - $F'_{V \rightarrow T} = A_{V \rightarrow T} \cdot (1 - u_{V \rightarrow T})$

---

## 🔗 Fase 6: Dempster-Shafer Evidential Fusion

**Tujuan:** Menggabungkan probabilitas subjektif (Opini) dari dua aliran menggunakan aturan Dempster-Shafer.

- **Step 6.1:** Definisikan fungsi untuk menggabungkan massa _belief_ dari aliran $T \rightarrow V$ dan $V \rightarrow T$.
  - Kalkulasi konstanta konflik (C) untuk normalisasi.
  - Kalkulasi $Belief_{Final}$ gabungan dengan rumus:
    $Bel_{Final,k} = \frac{1}{1-C}(b_{1,k} \cdot b_{2,k} + b_{1,k} \cdot u_2 + u_1 \cdot b_{2,k})$
- **Step 6.2:** Hasilkan prediksi akhir (kelas dengan massa _belief_ tertinggi).

---

## 🎯 Fase 7: Joint Loss Function & Optimizer

**Tujuan:** Mendefinisikan fungsi objektif untuk melatih klasifikasi sekaligus mengalibrasi metrik ketidakpastian.

- **Step 7.1:** Implementasikan EDL _Expected Mean Square Error_ (atau EDL Cross-Entropy) berbasis distribusi Dirichlet.
- **Step 7.2:** Tambahkan _Kullback-Leibler (KL) Divergence regularization term_ untuk mencegah model memberikan nilai evidensi tinggi pada data ber-_noise_.
  - $\mathcal{L}_{Total} = \mathcal{L}_{Classification} + \lambda_t \mathcal{L}_{KL}$
  - _Catatan:_ Gunakan _annealing step_ ($\lambda_t$) agar regularisasi KL meningkat secara perlahan seiring bertambahnya _epoch_.
- **Step 7.3:** Definisikan _Optimizer_ (misal: AdamW) dengan parameter _Learning Rate_ dan _Weight Decay_.

---

## 📈 Fase 8: Training Loop & Evaluasi

**Tujuan:** Melatih dan memvalidasi performa model SOTA.

- **Step 8.1:** Buat _loop training_ standar PyTorch (Forward pass, hitung _Loss_, Backward pass, _Optimizer step_).
- **Step 8.2:** Evaluasi model pada _Validation set_.
  - Gunakan metrik: _Accuracy, F1-Score (Macro)_, dan pantau rata-rata skor _Uncertainty_ pada data yang salah klasifikasi.
- **Step 8.3:** Simpan (_save_) bobot model terbaik berdasarkan F1-Score pada metrik validasi.
