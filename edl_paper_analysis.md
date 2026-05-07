# Analisis Paper: Evidential Deep Learning to Quantify Classification Uncertainty
**Penulis**: Murat Sensoy, Lance Kaplan, Melih Kandemir (NeurIPS 2018)

---

## 1. Ringkasan Inti (Core Insight)

Paper ini mengusulkan pendekatan **Evidential Deep Learning (EDL)** untuk mengkuantifikasi ketidakpastian (*uncertainty*) pada klasifikasi deep learning. Alih-alih menggunakan softmax yang hanya menghasilkan *point estimate* dari probabilitas kelas, EDL menempatkan **distribusi Dirichlet** di atas probabilitas kelas dan mempelajari parameter-parameter distribusi tersebut melalui neural network.

### Masalah yang Dipecahkan
- Softmax pada neural network standar menghasilkan **probabilitas tinggi bahkan untuk prediksi yang salah** (overconfident)
- Tidak ada cara bawaan untuk membedakan antara **ketidakpastian epistemic** (kurang pengetahuan) dan **ketidakpastian aleatory** (noise data)
- Model standar tidak bisa mengatakan "saya tidak tahu" (*I don't know*)

---

## 2. Fondasi Teoritis

### 2.1 Dempster-Shafer Theory of Evidence (DST)
- Generalisasi teori Bayesian ke probabilitas subjektif
- Menggunakan **belief mass** yang di-assign ke subset dari *frame of discernment*
- Memungkinkan ekspresi eksplisit "saya tidak tahu" dengan assign belief ke seluruh frame

### 2.2 Subjective Logic (SL)
Memformalisasi DST sebagai **Distribusi Dirichlet**:

- Untuk K kelas, ada K+1 nilai mass: belief mass `b_k` untuk setiap kelas k dan overall uncertainty `u`
- Constraint: `u + Σ b_k = 1`

**Formulasi evidence → belief & uncertainty:**

```
b_k = e_k / S    dan    u = K / S
```

di mana:
- `e_k ≥ 0` = evidence untuk kelas k
- `S = Σ(e_i + 1)` = Dirichlet strength
- `α_k = e_k + 1` = parameter Dirichlet

**Properti kunci:**
- Saat tidak ada evidence (semua `e_k = 0`): `u = 1` (total uncertainty)
- Saat evidence tinggi untuk satu kelas: uncertainty rendah, belief tinggi

### 2.3 Distribusi Dirichlet
```
D(p|α) = (1/B(α)) × Π p_i^(α_i - 1)    untuk p ∈ S_K
```

- Expected probability: `p̂_k = α_k / S`
- Distribusi Dirichlet memodelkan **probabilitas orde kedua** (distribusi atas distribusi probabilitas)

---

## 3. Arsitektur Model EDL

### 3.1 Modifikasi Neural Network
Perubahan arsitektur sangat **minimal** dari neural network standar:

1. **Hapus softmax layer** di output
2. **Ganti dengan activation layer** (ReLU) untuk menjamin output non-negatif
3. Output network = **evidence vector** `f(x_i|Θ)`
4. Parameter Dirichlet: `α_i = f(x_i|Θ) + 1`
5. Prediksi probabilitas: `p̂ = α_i / S_i`

### 3.2 Diagram Alur
```
Input → [Neural Network Backbone] → ReLU (non-negative) → Evidence e
                                                              ↓
                                                     α = e + 1
                                                              ↓
                                                     D(p|α) → belief mass b_k, uncertainty u
```

---

## 4. Loss Function

### 4.1 Tiga Opsi Loss Function yang Dibahas

#### Opsi 1: Negative Log Marginal Likelihood (Eq. 3)
```
L_i(Θ) = Σ_j y_ij × (log(S_i) - log(α_ij))
```
- Mengintegralkan keluar probabilitas kelas dari likelihood × prior Dirichlet

#### Opsi 2: Cross-Entropy Bayes Risk (Eq. 4)
```
L_i(Θ) = Σ_j y_ij × (ψ(S_i) - ψ(α_ij))
```
- ψ = digamma function
- Minimisasi Bayes risk terhadap cross-entropy loss

#### Opsi 3: Sum-of-Squares Bayes Risk (Eq. 5) — **YANG DIGUNAKAN**
```
L_i = Σ_j (y_ij - p̂_ij)² + p̂_ij(1 - p̂_ij)/(S_i + 1)
     = L_err + L_var
```
- `L_err` = squared error antara prediksi dan one-hot label
- `L_var` = variance dari prediksi Dirichlet (berfungsi sebagai regularizer)

### 4.2 KL Divergence Regularizer
```
L(Θ) = Σ L_i(Θ) + λ_t × Σ KL[D(p_i|α̃_i) || D(p_i|⟨1,...,1⟩)]
```
- `λ_t = min(1.0, t/10)` = annealing coefficient (bertambah seiring epoch)
- `α̃_i = y_i + (1 - y_i) ⊙ α_i` = parameter setelah removal non-misleading evidence
- Tujuan: penalize deviation dari state "I don't know" (uniform Dirichlet) yang tidak berkontribusi pada data fit
- Annealing mencegah konvergensi prematur ke uniform distribution

### 4.3 Tiga Proposisi Kunci (Properti Loss)
1. **Proposisi 1**: Variance term < Error term (`L_var < L_err`) → variance berfungsi sebagai regularizer
2. **Proposisi 2**: Error menurun saat evidence bertambah untuk kelas benar, naik saat evidence berkurang
3. **Proposisi 3**: Error menurun saat evidence dihapus dari parameter Dirichlet terbesar yang bukan kelas benar → model belajar menghapus misleading evidence

---

## 5. Hasil Eksperimen

### 5.1 Setup
- Arsitektur: LeNet standar dengan ReLU
- Dataset: MNIST, CIFAR-5/10
- Baseline: L2 (softmax standar), Dropout, Deep Ensemble, FFG, MNFG

### 5.2 Hasil Utama
| Benchmark | Temuan |
|-----------|--------|
| **Klasifikasi** | EDL setara dengan baseline (MNIST 99.3%, CIFAR5 83%) |
| **OOD Detection** | EDL **secara signifikan lebih baik** — entropy tinggi untuk data OOD (notMNIST) |
| **Adversarial Robustness** | EDL **paling robust** — uncertainty meningkat tajam saat adversarial perturbation ditambahkan |
| **Uncertainty Threshold** | Akurasi meningkat menjadi 100% saat prediksi di atas uncertainty threshold dibuang |

### 5.3 Keunggulan EDL vs Baseline
- **Single forward pass** (vs MC Dropout yang butuh multiple passes)
- **Tidak perlu ensemble** (vs Deep Ensemble yang butuh multiple model)
- **Uncertainty decomposition** langsung tersedia (epistemic vs aleatory)
- Komputasi **jauh lebih efisien**

---

## 6. Implikasi untuk Tesis: Multimodal Sentiment Analysis

### 6.1 Relevansi EDL untuk Multimodal
- EDL memberikan **uncertainty per-modalitas** yang bisa digunakan untuk weighting
- Uncertainty bisa digunakan untuk **mendeteksi modalitas yang noisy/tidak informatif**
- Evidence dari setiap modalitas bisa di-**combine** menggunakan Dempster's rule of combination
- Cocok untuk **adaptive fusion** — modalitas dengan uncertainty tinggi diberi bobot rendah

### 6.2 Potensi Integrasi dengan Co-Attention
- Co-attention menghasilkan **attended features** per modalitas
- EDL bisa diterapkan **setelah** co-attention untuk mengkuantifikasi uncertainty dari attended features
- Uncertainty bisa digunakan sebagai **gate** atau **selection mechanism** untuk memilih informasi mana yang dipercaya

### 6.3 Uncertainty-Aware Selection Module
- Menggunakan uncertainty dari EDL untuk **memfilter** atau **meweighting** representasi multimodal
- Modalitas/region dengan uncertainty tinggi → dampened/filtered
- Modalitas/region dengan uncertainty rendah → amplified/selected
