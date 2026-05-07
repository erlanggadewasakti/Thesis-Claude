# Deep Dive: Loss Functions pada Evidential Deep Learning

---

## 0. Prasyarat: Memahami Notasi & Setup

Sebelum masuk ke loss function, kita harus memahami setup dasar EDL:

### Apa yang Diprediksi Network?
Pada neural network standar, output layer menggunakan **softmax** untuk menghasilkan probabilitas kelas. Di EDL:

```
Neural Network → ReLU → evidence vector e = f(x_i|Θ)    (semua ≥ 0)
                                ↓
                         α_i = e_i + 1    (parameter Dirichlet, semua ≥ 1)
                                ↓
                         S_i = Σ α_ij     (Dirichlet strength)
                                ↓
                         p̂_ij = α_ij / S_i    (expected probability kelas j)
```

### Mengapa Distribusi Dirichlet?
Softmax menghasilkan **satu set probabilitas** (point estimate). Dirichlet menghasilkan **distribusi atas semua kemungkinan set probabilitas**. Ini artinya kita tidak hanya tahu "kelas A punya probabilitas 0.8", tapi juga **seberapa yakin kita tentang angka 0.8 itu sendiri**.

### Variabel-variabel kunci:
| Simbol | Makna |
|--------|-------|
| `x_i` | Input sample ke-i |
| `y_i` | One-hot label (y_ij = 1 untuk kelas benar j) |
| `e_ij` | Evidence untuk kelas j dari sample i (output network, ≥ 0) |
| `α_ij` | Parameter Dirichlet = `e_ij + 1` (≥ 1) |
| `S_i` | Dirichlet strength = `Σ_j α_ij` |
| `p̂_ij` | Expected probability = `α_ij / S_i` |
| `K` | Jumlah kelas |
| `Θ` | Parameter network |

---

## 1. Loss Function #1: Negative Log Marginal Likelihood (Type II ML)

### Formulasi (Eq. 3)

```
L_i(Θ) = Σ_j  y_ij × (log(S_i) − log(α_ij))
```

### Derivasi: Dari Mana Rumus Ini Berasal?

**Ide**: Kita punya likelihood data `Mult(y_i|p_i)` dan prior Dirichlet `D(p_i|α_i)`. Kita **integralkan keluar** (marginalize) probabilitas kelas `p_i`:

```
L_i = −log ∫ [ Π_j p_ij^y_ij ] × [ (1/B(α)) × Π_j p_ij^(α_ij−1) ] dp_i
```

Ini adalah integral dari produk dua distribusi atas simplex:
- `Π_j p_ij^y_ij` → likelihood multinomial (hanya kelas benar j yang punya y_ij = 1)
- `(1/B(α)) × Π_j p_ij^(α_ij−1)` → prior Dirichlet

Mengombinasikan keduanya:

```
∫ Π_j p_ij^(y_ij + α_ij − 1) dp_i × (1/B(α))
```

Integral ini punya bentuk **Beta function**:

```
∫ Π_j p_ij^(y_ij + α_ij − 1) dp_i = B(y_i + α_i)
```

Sehingga:

```
L_i = −log [ B(y_i + α_i) / B(α_i) ]
```

Karena `y_i` adalah one-hot (hanya y_ij = 1 untuk kelas benar j), ini menyederhanakan menjadi:

```
L_i = log(S_i) − log(α_ij)    (hanya untuk j = kelas benar)
```

Atau dalam bentuk sum:

```
L_i = Σ_j y_ij × (log(S_i) − log(α_ij))
```

### Interpretasi Intuitif

- **`log(S_i)`**: Semakin besar total Dirichlet strength → loss semakin besar. Ini **mendorong model untuk tidak "overgenerate" evidence secara keseluruhan**.
- **`−log(α_ij)`**: Semakin besar evidence untuk kelas benar → loss semakin kecil. Ini **mendorong model menghasilkan evidence tinggi untuk kelas yang benar**.
- **Trade-off**: Model harus meningkatkan `α_ij` (kelas benar) **tanpa** meningkatkan `S_i` terlalu banyak, artinya evidence untuk kelas lain harus tetap rendah.

### Analogi
Ini mirip dengan **negative log-likelihood** standar, tetapi alih-alih langsung menggunakan `p`, kita mengintegralkan atas semua kemungkinan `p` yang konsisten dengan distribusi Dirichlet kita. Ini disebut **Type II Maximum Likelihood** dalam literatur Bayesian.

### Kelemahan (diobservasi empiris)
- Cenderung menghasilkan **belief mass yang berlebihan** (excessively high)
- Performa **kurang stabil** dibanding Loss #3

---

## 2. Loss Function #2: Cross-Entropy Bayes Risk

### Formulasi (Eq. 4)

```
L_i(Θ) = Σ_j  y_ij × (ψ(S_i) − ψ(α_ij))
```

di mana `ψ(·)` adalah **digamma function** (turunan dari log gamma function: `ψ(x) = d/dx ln Γ(x)`).

### Derivasi: Bayes Risk dengan Cross-Entropy Loss

**Ide**: Alih-alih mengintegralkan likelihood × prior (seperti Loss #1), kita langsung **menghitung expected value dari cross-entropy loss** terhadap distribusi Dirichlet.

Cross-entropy loss standar untuk satu sample:

```
ℓ(y_i, p_i) = −Σ_j y_ij × log(p_ij)
```

**Bayes risk** = ekspektasi loss terhadap distribusi posterior atas `p`:

```
L_i = E_{p ~ Dir(α)} [ −Σ_j y_ij × log(p_ij) ]
    = −Σ_j y_ij × E_{p ~ Dir(α)} [ log(p_ij) ]
```

Untuk distribusi Dirichlet, kita punya properti:

```
E[log(p_ij)] = ψ(α_ij) − ψ(S_i)
```

> **Mengapa?** Ini adalah properti standar distribusi Dirichlet. Ekspektasi dari log komponen ke-j dari variabel random Dirichlet sama dengan digamma dari parameter ke-j dikurangi digamma dari jumlah semua parameter.

Substitusi:

```
L_i = −Σ_j y_ij × (ψ(α_ij) − ψ(S_i))
    = Σ_j y_ij × (ψ(S_i) − ψ(α_ij))
```

### Interpretasi Intuitif

- Strukturnya **sangat mirip** dengan Loss #1: `log(S)−log(α)` vs `ψ(S)−ψ(α)`
- Perbedaannya: digamma function `ψ(x) ≈ log(x) − 1/(2x)` untuk x besar
- Untuk α besar, kedua loss **hampir identik**
- Untuk α kecil, digamma memberikan **gradien yang berbeda** (lebih smooth)

### Perbedaan Kunci dengan Loss #1

| Aspek | Loss #1 (Marginal Likelihood) | Loss #2 (CE Bayes Risk) |
|-------|-------------------------------|-------------------------|
| **Pendekatan** | Integralkan likelihood × prior | Expected value dari loss |
| **Fungsi** | `log(S) − log(α)` | `ψ(S) − ψ(α)` |
| **Classifier type** | Bayes classifier (PAC-learning) | Gibbs classifier |
| **Behaviour** | Lebih agresif | Sedikit lebih smooth |

> **Catatan**: Dalam terminologi PAC-learning, **Bayes classifier** memilih kelas yang memaksimalkan expected probability (optimal secara deterministik), sedangkan **Gibbs classifier** men-sample kelas dari distribusi posterior (stochastic). Loss #1 sesuai dengan Bayes classifier, Loss #2 dan #3 sesuai dengan Gibbs classifier.

### Kelemahan (diobservasi empiris)
- Sama seperti Loss #1: cenderung menghasilkan **belief mass berlebihan**
- Performa **kurang stabil** dibanding Loss #3

---

## 3. Loss Function #3: Sum of Squares (SSE) Bayes Risk — **YANG DIPILIH**

### Formulasi (Eq. 5)

```
L_i(Θ) = ∫ ||y_i − p_i||²₂ × Dir(p_i|α_i) dp_i
```

### Derivasi Lengkap

**Step 1**: Ekspansi L2-norm:

```
||y_i − p_i||²₂ = Σ_j (y_ij − p_ij)²
```

**Step 2**: Hitung ekspektasi terhadap Dirichlet:

```
L_i = Σ_j E[(y_ij − p_ij)²]
    = Σ_j E[y²_ij − 2·y_ij·p_ij + p²_ij]
    = Σ_j (y²_ij − 2·y_ij·E[p_ij] + E[p²_ij])
```

(`y_ij` adalah konstanta, bisa dikeluarkan dari ekspektasi)

**Step 3**: Substitusi momen-momen distribusi Dirichlet:

Untuk distribusi Dirichlet `Dir(α)`:
- **Momen pertama**: `E[p_ij] = α_ij / S_i`
- **Momen kedua**: `E[p²_ij] = E[p_ij]² + Var(p_ij)`

Variance Dirichlet:
```
Var(p_ij) = α_ij(S_i − α_ij) / (S_i²(S_i + 1))
```

Maka:
```
E[p²_ij] = (α_ij/S_i)² + α_ij(S_i − α_ij) / (S_i²(S_i + 1))
```

**Step 4**: Substitusi kembali dan simplifikasi:

```
L_i = Σ_j [ (y_ij − α_ij/S_i)²  +  α_ij(S_i − α_ij) / (S_i²(S_i + 1)) ]
         \_________________/     \________________________________/
              L_err_ij                       L_var_ij
```

Atau dalam bentuk lebih ringkas menggunakan `p̂_ij = α_ij/S_i`:

```
L_i = Σ_j [ (y_ij − p̂_ij)²  +  p̂_ij(1 − p̂_ij) / (S_i + 1) ]
```

### Dekomposisi: Error + Variance

Loss ini secara natural terdekomposisi menjadi dua komponen:

#### Komponen 1: `L_err` (Prediction Error)
```
L_err_ij = (y_ij − p̂_ij)²
```
- **Squared error** antara label one-hot dan predicted probability
- Mendorong model untuk **memprediksi dengan benar**
- Mirip dengan MSE loss standar

#### Komponen 2: `L_var` (Prediction Variance)
```
L_var_ij = p̂_ij(1 − p̂_ij) / (S_i + 1)
```
- **Variance** dari distribusi Dirichlet untuk komponen ke-j
- Mendorong model untuk **mengurangi ketidakpastian** prediksinya
- Semakin besar `S_i` (total evidence) → variance semakin kecil → loss semakin kecil
- TETAPI hanya jika evidence tersebut juga mengurangi error (Proposisi 1)

### Mengapa Loss Ini Dipilih? Tiga Proposisi Kunci

#### Proposisi 1: Variance Selalu Lebih Kecil dari Error
```
L_var_ij < L_err_ij    (untuk semua α_ij ≥ 1)
```

**Implikasi**: Model **memprioritaskan data fit (mengurangi error)** daripada variance reduction. Variance berfungsi sebagai **regularizer yang gentle** — ia mempengaruhi training tapi tidak mendominasi.

**Bukti intuitif**: 
- `L_var = α_ij(S_i − α_ij) / (S_i²(S_i + 1))`
- Denominator punya faktor `(S_i + 1)` tambahan, membuatnya selalu lebih kecil

#### Proposisi 2: Evidence Benar → Error Menurun
Untuk kelas benar j, menambah evidence ke `α_ij`:
- `L_err` **pasti menurun** (karena `p̂_ij = α_ij/S_i` mendekati 1)
- Menghapus evidence dari `α_ij` → `L_err` **pasti naik**

**Implikasi**: Loss ini memiliki **gradient yang benar** — selalu mendorong model menghasilkan lebih banyak evidence untuk kelas yang benar.

#### Proposisi 3: Menghapus Misleading Evidence → Error Menurun
Menghapus evidence dari `α_il` (kelas salah l ≠ j, yang punya evidence terbesar):
- `L_err` **pasti menurun**

**Implikasi**: Model secara natural belajar **menghapus evidence yang misleading** dari kelas-kelas yang salah. Ini adalah properti **learned loss attenuation**.

### Contoh Numerik: Loss #3 Beraksi

Misalkan K=3 (3 kelas), sample benar = kelas 1 → `y = [1, 0, 0]`

**Skenario A: Model yakin dan benar**
```
evidence e = [10, 0, 0]  →  α = [11, 1, 1]  →  S = 13
p̂ = [11/13, 1/13, 1/13] = [0.846, 0.077, 0.077]

L_err = (1−0.846)² + (0−0.077)² + (0−0.077)² = 0.0237 + 0.0059 + 0.0059 = 0.0355
L_var = 0.846×0.154/14 + 0.077×0.923/14 + 0.077×0.923/14 = 0.0093 + 0.0051 + 0.0051 = 0.0195
Total L = 0.0355 + 0.0195 = 0.0550  ✓ Rendah!
```

**Skenario B: Model tidak yakin (evidence rendah)**
```
evidence e = [1, 0, 0]  →  α = [2, 1, 1]  →  S = 4
p̂ = [2/4, 1/4, 1/4] = [0.5, 0.25, 0.25]

L_err = (1−0.5)² + (0−0.25)² + (0−0.25)² = 0.25 + 0.0625 + 0.0625 = 0.375
L_var = 0.5×0.5/5 + 0.25×0.75/5 + 0.25×0.75/5 = 0.05 + 0.0375 + 0.0375 = 0.125
Total L = 0.375 + 0.125 = 0.500  ⚠️ Cukup tinggi
```

**Skenario C: Model yakin tapi SALAH**
```
evidence e = [0, 10, 0]  →  α = [1, 11, 1]  →  S = 13
p̂ = [1/13, 11/13, 1/13] = [0.077, 0.846, 0.077]

L_err = (1−0.077)² + (0−0.846)² + (0−0.077)² = 0.852 + 0.716 + 0.006 = 1.574
L_var = 0.077×0.923/14 + 0.846×0.154/14 + 0.077×0.923/14 = 0.005 + 0.009 + 0.005 = 0.019
Total L = 1.574 + 0.019 = 1.593  ❌ Sangat tinggi! (bagus, karena model salah)
```

**Skenario D: Tidak ada evidence sama sekali (total uncertainty)**
```
evidence e = [0, 0, 0]  →  α = [1, 1, 1]  →  S = 3
p̂ = [1/3, 1/3, 1/3] = [0.333, 0.333, 0.333]
uncertainty u = K/S = 3/3 = 1.0  (total uncertainty)

L_err = (1−0.333)² + (0−0.333)² + (0−0.333)² = 0.444 + 0.111 + 0.111 = 0.666
L_var = 0.333×0.667/4 + 0.333×0.667/4 + 0.333×0.667/4 = 0.0556×3 = 0.167
Total L = 0.666 + 0.167 = 0.833
```

> [!IMPORTANT]
> **Perhatikan**: Skenario D (total uncertainty) memiliki loss **lebih rendah** daripada Skenario C (salah tapi yakin). Ini berarti model EDL lebih memilih mengatakan "saya tidak tahu" daripada "saya yakin tapi salah"! Ini adalah properti yang sangat diinginkan.

---

## 4. KL Divergence Regularizer

### Masalah yang Dipecahkan

Loss #3 saja belum cukup. Masalahnya: model bisa menghasilkan **misleading evidence** untuk kelas-kelas yang salah, selama evidence untuk kelas benar tetap lebih tinggi. Contoh:

```
evidence = [50, 30, 20]  →  α = [51, 31, 21]  →  S = 103
p̂ = [0.495, 0.301, 0.204]

Prediksi benar (kelas 1), TAPI ada banyak evidence "sampah" untuk kelas 2 dan 3
```

Kita ingin idealnya: `evidence = [50, 0, 0]` — evidence hanya untuk kelas yang benar.

### Formulasi

```
L_total(Θ) = Σ_i L_i(Θ) + λ_t × Σ_i KL[Dir(p_i|α̃_i) ‖ Dir(p_i|⟨1,...,1⟩)]
```

### Komponen-komponen:

#### α̃ (Dirichlet parameter setelah removal evidence benar)
```
α̃_i = y_i + (1 − y_i) ⊙ α_i
```

**Apa yang terjadi?** Untuk kelas benar j (y_ij = 1):
```
α̃_ij = 1    (evidence kelas benar di-reset ke 1, karena ini bukan misleading)
```

Untuk kelas salah k ≠ j (y_ik = 0):
```
α̃_ik = α_ik  (evidence kelas salah dipertahankan, karena INI yang misleading)
```

**Contoh**:
```
y = [1, 0, 0],  α = [51, 31, 21]
α̃ = [1, 31, 21]  ← evidence kelas benar dihapus, sisanya dipertahankan
```

#### Dir(p|⟨1,...,1⟩) — Target: Uniform Dirichlet
- Distribusi Dirichlet dengan semua parameter = 1
- Ini adalah state **"saya tidak tahu"** — tidak ada evidence sama sekali
- KL divergence mengukur **seberapa jauh** distribusi saat ini dari state "tidak tahu"

#### KL Divergence antara dua Dirichlet
```
KL[Dir(α̃) ‖ Dir(1)] = log[Γ(Σ α̃_k) / (Γ(K) × Π Γ(α̃_k))]
                       + Σ_k (α̃_k − 1) × [ψ(α̃_k) − ψ(Σ α̃_j)]
```

di mana `Γ(·)` = gamma function, `ψ(·)` = digamma function.

### λ_t: Annealing Coefficient

```
λ_t = min(1.0, t/10)
```

di mana `t` = epoch saat ini.

**Mengapa annealing?**

| Epoch | λ_t | Efek |
|-------|-----|------|
| 0 | 0.0 | KL tidak berpengaruh → model bebas explore |
| 1 | 0.1 | KL mulai sedikit berpengaruh |
| 5 | 0.5 | KL berpengaruh setengah |
| 10+ | 1.0 | KL berpengaruh penuh |

**Rasionalnya**: Di awal training, banyak sample yang masih misclassified. Jika KL langsung berpengaruh penuh, model akan langsung converge ke uniform distribution (total uncertainty) untuk sample-sample ini. Padahal, sample yang tadinya misclassified **mungkin bisa benar** di epoch selanjutnya. Annealing memberikan "kesempatan kedua" bagi model.

### Efek Gabungan KL Regularizer

```
Tanpa KL: evidence = [50, 30, 20] → model puas (prediksi benar)
Dengan KL: evidence = [50, 30, 20] → KL penalty tinggi karena α̃ = [1, 31, 21]
                                       jauuuh dari uniform [1, 1, 1]
           → model didorong ke evidence = [50, 0, 0] → α̃ = [1, 1, 1] → KL = 0!
```

> [!TIP]
> **Insight kunci**: KL regularizer hanya **menghukum evidence yang misleading** (untuk kelas-kelas salah). Evidence untuk kelas benar di-remove sebelum KL dihitung, sehingga model tetap bebas menghasilkan evidence sebanyak apapun untuk kelas benar tanpa penalty.

---

## 5. Perbandingan Keseluruhan Tiga Loss Functions

| Aspek | Loss #1 (NLL) | Loss #2 (CE Bayes Risk) | Loss #3 (SSE Bayes Risk) |
|-------|---------------|------------------------|-------------------------|
| **Formula** | `Σ y(log S − log α)` | `Σ y(ψ(S) − ψ(α))` | `Σ (y−p̂)² + Var(p)` |
| **Derivasi** | Marginalize likelihood | E[cross-entropy] | E[squared error] |
| **Classifier** | Bayes | Gibbs | Gibbs |
| **Interpretability** | Medium | Low (digamma) | **Tinggi** (error + variance) |
| **Stabilitas** | Kurang stabil | Kurang stabil | **Paling stabil** |
| **Belief mass** | Cenderung excessive | Cenderung excessive | **Terkontrol** |
| **Dipilih?** | ❌ | ❌ | ✅ |

### Mengapa Loss #3 Superior?

1. **Decomposable**: Terurai jelas menjadi error (data fit) + variance (uncertainty)
2. **Self-regulating**: Proposisi 1 menjamin variance tidak mendominasi error
3. **Properti gradient yang baik**: Proposisi 2 & 3 menjamin gradient mendorong ke arah yang benar
4. **Empiris lebih stabil**: Tidak menghasilkan belief mass yang berlebihan

---

## 6. Total Loss: Rangkuman Akhir

```
L_total = Σᵢ [ Σⱼ (yᵢⱼ − p̂ᵢⱼ)² + p̂ᵢⱼ(1−p̂ᵢⱼ)/(Sᵢ+1) ]    ← SSE Bayes Risk
        + λₜ × Σᵢ KL[ Dir(α̃ᵢ) ‖ Dir(1) ]                   ← KL Regularizer
```

**Apa yang didorong oleh setiap komponen:**

```
┌─────────────────────────┬───────────────────────────────────────────────┐
│ Komponen                │ Mendorong model untuk...                     │
├─────────────────────────┼───────────────────────────────────────────────┤
│ L_err (squared error)   │ Memprediksi kelas yang benar                │
│ L_var (variance)        │ Menghasilkan evidence yang cukup (yakin)     │
│ KL (regularizer)        │ Menghapus misleading evidence (kelas salah)  │
│ λ_t (annealing)         │ Memberikan waktu explore sebelum converge    │
└─────────────────────────┴───────────────────────────────────────────────┘
```

---

## 7. Relevansi untuk Tesis Multimodal

Untuk tesis Anda tentang **Uncertainty-Aware Selection + EDL + Co-Attention**:

### Loss Function yang Direkomendasikan
Gunakan **Loss #3 (SSE Bayes Risk) + KL Regularizer** karena:

1. **Per-modalitas EDL**: Setiap branch modalitas (teks, gambar, audio) menghasilkan evidence vector → `L_var` memberikan uncertainty per modalitas
2. **Uncertainty-aware weighting**: `u = K/S` dari setiap modalitas bisa digunakan untuk menentukan bobot fusi
3. **Dekomposisi yang jelas**: Error dan variance terpisah, memungkinkan analisis kontribusi setiap modalitas
4. **KL regularizer**: Mencegah modalitas yang noisy menghasilkan misleading evidence yang bisa meracuni fusi

### Potensi Modifikasi untuk Multimodal
- **Separate KL per modalitas**: Setiap modalitas punya KL regularizer sendiri
- **Joint KL setelah fusi**: KL divergence diterapkan pada distribusi Dirichlet yang sudah di-fusi
- **Weighted loss**: Bobot loss tiap modalitas bisa di-scale berdasarkan reliability
