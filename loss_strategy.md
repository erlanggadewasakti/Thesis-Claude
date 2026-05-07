# Strategi Loss Function: 3-Head EDL + Dempster

---

## 1. Struktur Total Loss

Untuk arsitektur revisi (3 head + Dempster), total loss terdiri dari **4 komponen EDL**:

```
L_total = L_text + L_image + L_coattn + L_final
```

Masing-masing menggunakan format yang sama:

```
L_head = Class-Weighted EDL Loss + λ_t × Class-Weighted KL Regularizer
```

### Mengapa 4 Komponen?

```
                                    Loss Signal
                                    ──────────
Head Text ──── α_t ──── L_text      → update BERT + text projection + text EDL head
Head Image ─── α_v ──── L_image     → update ResNet + image projection + image EDL head
Head CoAttn ── α_c ──── L_coattn    → update co-attention layers + coattn EDL head
Dempster ───── α_f ──── L_final     → update SEMUA (gradient flow ke seluruh network)
```

**L_final adalah yang paling penting** — ini memastikan seluruh pipeline di-optimize end-to-end. Tiga loss lainnya berfungsi sebagai **auxiliary supervision** yang:
1. Mencegah overfitting (regularisasi implisit dari multi-task)
2. Memastikan setiap head menghasilkan prediksi bermakna secara independen
3. Memberikan gradient signal yang lebih kuat ke setiap branch

---

## 2. Detail Setiap Komponen Loss

### 2.1 Class-Weighted EDL Loss (SSE Bayes Risk)

Untuk satu sample `i` dengan label kelas benar `j`:

```
L_EDL(α_i, y_i) = Σ_k  w_k × [(y_ik − p̂_ik)² + p̂_ik(1 − p̂_ik)/(S_i + 1)]
```

di mana:
- `p̂_ik = α_ik / S_i` (expected probability kelas k)
- `S_i = Σ α_ik` (Dirichlet strength)
- `w_k` = **class weight** untuk kelas k

#### Bagaimana Menghitung Class Weight?

Dari data Anda: Neg=204, Neu=70, Pos=403. Total N=677.

```python
# Inverse frequency weighting
w_neg = N / (K × N_neg) = 677 / (3 × 204) = 1.106
w_neu = N / (K × N_neu) = 677 / (3 × 70)  = 3.224
w_pos = N / (K × N_pos) = 677 / (3 × 403) = 0.560

# Setelah normalisasi agar rata-rata = 1:
# w = [1.106, 3.224, 0.560] → sum = 4.89, mean = 1.63
# Normalized: w = [0.678, 1.977, 0.343] × (3/3) → keep as is
```

> [!IMPORTANT]
> Class weight **bukan** mengalikan seluruh loss per sample. Class weight mengalikan **kontribusi per kelas** dalam sum of squares. Ini penting karena EDL loss dihitung per kelas, bukan per sample.

#### Implementasi yang Benar

```python
def edl_loss(alpha, y_onehot, class_weights):
    """
    alpha: (batch, K) — Dirichlet parameters
    y_onehot: (batch, K) — one-hot labels
    class_weights: (K,) — inverse frequency weights
    """
    S = alpha.sum(dim=1, keepdim=True)        # (batch, 1)
    p_hat = alpha / S                          # (batch, K)
    
    # Error term: (y - p̂)²
    err = (y_onehot - p_hat) ** 2              # (batch, K)
    
    # Variance term: p̂(1-p̂) / (S+1)
    var = p_hat * (1 - p_hat) / (S + 1)       # (batch, K)
    
    # Per-class weighted loss
    loss_per_class = class_weights * (err + var)   # (batch, K)
    
    # Sum over classes, mean over batch
    loss = loss_per_class.sum(dim=1).mean()
    
    return loss
```

### 2.2 Class-Weighted KL Regularizer

KL regularizer mendorong evidence **kelas salah** ke nol. Yang kritis: class weight juga harus diterapkan di sini.

```python
def kl_regularizer(alpha, y_onehot, class_weights, epoch, annealing_epochs=10):
    """
    Hanya menghukum evidence untuk kelas yang SALAH.
    """
    # Annealing coefficient
    lambda_t = min(1.0, epoch / annealing_epochs)
    
    # Remove evidence dari kelas benar (jangan dihukum)
    # α̃ = y + (1-y) ⊙ α  →  kelas benar: α̃=1, kelas salah: α̃=α
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    
    # KL divergence: Dir(α̃) vs Dir(1,1,...,1)
    K = alpha.shape[1]
    ones = torch.ones_like(alpha)
    
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    
    # KL computation
    ln_B = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
    ln_B_uni = torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    
    dg0 = torch.digamma(S_tilde)
    dg1 = torch.digamma(alpha_tilde)
    
    # Per-class KL contribution, weighted
    kl_per_class = (alpha_tilde - 1) * (dg1 - dg0)          # (batch, K)
    kl_weighted = (class_weights * kl_per_class).sum(dim=1, keepdim=True)
    
    kl = kl_weighted + ln_B + ln_B_uni
    
    return lambda_t * kl.mean()
```

### 2.3 Total Loss Per Head

```python
def head_loss(alpha, y_onehot, class_weights, epoch):
    return edl_loss(alpha, y_onehot, class_weights) \
         + kl_regularizer(alpha, y_onehot, class_weights, epoch)
```

### 2.4 Total Loss Keseluruhan

```python
def total_loss(alpha_t, alpha_v, alpha_c, alpha_final, y_onehot, class_weights, epoch):
    L_text   = head_loss(alpha_t, y_onehot, class_weights, epoch)
    L_image  = head_loss(alpha_v, y_onehot, class_weights, epoch)
    L_coattn = head_loss(alpha_c, y_onehot, class_weights, epoch)
    L_final  = head_loss(alpha_final, y_onehot, class_weights, epoch)
    
    # Weighted sum
    # L_final paling penting (end-to-end), auxiliary heads sedikit lebih rendah
    return 0.5 * L_text + 0.5 * L_image + 0.5 * L_coattn + 1.0 * L_final
```

> [!TIP]
> Auxiliary heads (`L_text`, `L_image`, `L_coattn`) diberi bobot 0.5 agar tidak mendominasi L_final. Ini bisa di-tune, tapi pastikan L_final selalu punya bobot tertinggi karena ini yang menentukan prediksi akhir.

---

## 3. Bagaimana Setiap Komponen Mendorong Update Bobot yang Optimal

### 3.1 L_err: Mendorong Prediksi Benar

```
∂L_err/∂α_j = −2(y_j − α_j/S) × (y_j·S − α_j) / S²
```

- Untuk kelas benar (y_j=1): gradient **mendorong α_j naik** (lebih banyak evidence)
- Untuk kelas salah (y_j=0): gradient **mendorong α_k turun** (kurangi misleading evidence)
- Class weight memperbesar gradient untuk neutral → **model dipaksa lebih memperhatikan neutral**

### 3.2 L_var: Mencegah Overconfidence (Anti-Overfit)

```
L_var = p̂(1−p̂) / (S+1)
```

- Saat S besar (banyak evidence) → L_var kecil → model "reward" karena yakin
- TAPI: model HANYA boleh yakin jika L_err juga kecil (Proposisi 1: L_var < L_err)
- Ini mencegah model **menghasilkan evidence besar secara sembarangan** hanya untuk terlihat yakin
- Efek anti-overfit: model tidak bisa sekadar menghafalkan training data dengan evidence besar

### 3.3 KL Regularizer: Menghapus Evidence Sampah

```
KL[Dir(α̃) ‖ Dir(1)] → mendorong α̃ → 1 untuk kelas salah
```

- α̃ untuk kelas benar = 1 (tidak dihukum)
- α̃ untuk kelas salah = α_k (DIHUKUM jika > 1)
- Efek: model **dipaksa menghasilkan evidence HANYA untuk kelas yang benar**
- Ini SANGAT membantu mengatasi overfitting karena model tidak bisa "hedge" dengan menyebarkan evidence ke semua kelas

### 3.4 Annealing λ_t: Mencegah Konvergensi Prematur

```
λ_t = min(1.0, epoch / 10)
```

| Epoch | λ_t | Efek pada Training |
|-------|-----|--------------------|
| 0-3 | 0.0-0.3 | Model bebas explore, boleh salah, boleh generate evidence untuk kelas salah |
| 4-7 | 0.4-0.7 | KL mulai menghukum misleading evidence, model mulai "membersihkan" |
| 8-10 | 0.8-1.0 | KL penuh aktif, model harus yakin HANYA pada kelas benar |
| 10+ | 1.0 | Full regularization — evidence bersih, uncertainty terkalibrasi |

**Tanpa annealing**: KL langsung penuh → model di epoch awal langsung collapse ke uniform → tidak pernah belajar → underfitting.

### 3.5 Class Weight: Mengatasi Imbalance

Tanpa class weight:
```
Model melihat: 403 positive, 204 negative, 70 neutral
→ Gradient dari positive mendominasi (6× lebih banyak dari neutral)
→ Model bias ke positive
```

Dengan class weight:
```
w_neu = 3.224 → gradient dari neutral diperkuat 3.2×
→ Satu sample neutral punya "suara" setara 3.2 sample positive
→ Model dipaksa memperhatikan neutral
```

---

## 4. Contoh Numerik: Forward Pass + Loss

K=3 kelas (neg=0, neu=1, pos=2), `class_weights = [1.106, 3.224, 0.560]`

### Skenario A: Model benar dan yakin (sample positive)

```
y = [0, 0, 1],  evidence e = [0.5, 0.3, 15]
α = [1.5, 1.3, 16]  →  S = 18.8
p̂ = [0.080, 0.069, 0.851]
u = 3/18.8 = 0.160 (rendah → bagus!)

L_err = 1.106×(0-0.080)² + 3.224×(0-0.069)² + 0.560×(1-0.851)²
      = 1.106×0.0064 + 3.224×0.0048 + 0.560×0.0222
      = 0.0071 + 0.0154 + 0.0124 = 0.0349

L_var = 1.106×0.080×0.920/19.8 + 3.224×0.069×0.931/19.8 + 0.560×0.851×0.149/19.8
      = 0.0041 + 0.0105 + 0.0036 = 0.0182

Total L_EDL = 0.0349 + 0.0182 = 0.0531  ✅ Rendah
```

### Skenario B: Model salah (sample neutral diprediksi positive)

```
y = [0, 1, 0],  evidence e = [0.5, 0.3, 15]
α = [1.5, 1.3, 16]  →  S = 18.8
p̂ = [0.080, 0.069, 0.851]

L_err = 1.106×(0-0.080)² + 3.224×(1-0.069)² + 0.560×(0-0.851)²
      = 0.0071 + 3.224×0.867 + 0.560×0.724
      = 0.0071 + 2.7952 + 0.4054 = 3.2077  ❌ SANGAT tinggi

L_var = 0.0041 + 0.0105 + 0.0036 = 0.0182

Total L_EDL = 3.2077 + 0.0182 = 3.2259
```

> [!IMPORTANT]
> **Perhatikan**: Karena `w_neu = 3.224`, loss untuk misklasifikasi neutral **diperkuat 3.2×**. Error `(1-0.069)²` = 0.867 dikalikan 3.224 menjadi **2.795**! Ini memaksa model menghasilkan gradient besar untuk memperbaiki prediksi neutral.

### Skenario C: Model tidak yakin (evidence rendah semua)

```
y = [0, 1, 0],  evidence e = [0.1, 0.2, 0.1]
α = [1.1, 1.2, 1.1]  →  S = 3.4
p̂ = [0.324, 0.353, 0.324]
u = 3/3.4 = 0.882 (tinggi → model bilang "I don't know")

L_err = 1.106×(0-0.324)² + 3.224×(1-0.353)² + 0.560×(0-0.324)²
      = 0.116 + 1.351 + 0.059 = 1.526

L_var = 1.106×0.324×0.676/4.4 + 3.224×0.353×0.647/4.4 + 0.560×0.324×0.676/4.4
      = 0.055 + 0.167 + 0.028 = 0.250

Total L_EDL = 1.526 + 0.250 = 1.776
```

**Bandingkan Skenario B vs C**:
- B (salah tapi yakin): **3.226** → gradient sangat besar → bobot di-update agresif
- C (tidak yakin): **1.776** → gradient lebih kecil → update lebih gentle

Ini berarti EDL **menghukum kesalahan yang overconfident jauh lebih berat** daripada ketidaktahuan yang jujur. Model belajar: "lebih baik tidak yakin daripada salah tapi yakin."

---

## 5. KL Regularizer: Contoh Numerik

Lanjutkan Skenario B (model salah prediksi neutral → positive):

```
y = [0, 1, 0],  α = [1.5, 1.3, 16]

# Remove evidence kelas benar (kelas 1, neutral)
α̃ = y + (1-y) ⊙ α = [0,1,0] + [1,0,1] ⊙ [1.5, 1.3, 16]
   = [0,1,0] + [1.5, 0, 16] = [1.5, 1, 16]

# KL[Dir(1.5, 1, 16) ‖ Dir(1, 1, 1)]
# α̃ - 1 = [0.5, 0, 15]  ← evidence kelas salah yang harus dihukum!

# Kelas 0 (neg): excess evidence = 0.5 → small penalty
# Kelas 1 (neu): excess evidence = 0 → NO penalty (kelas benar)
# Kelas 2 (pos): excess evidence = 15 → HUGE penalty!
```

KL regularizer akan menghasilkan **penalty besar** karena ada 15 unit evidence yang misleading untuk kelas positive, padahal kelas benar adalah neutral.

Efek pada training: gradient dari KL akan **mendorong network untuk mengurangi evidence positive** untuk sample ini, sehingga di epoch berikutnya, model menghasilkan lebih sedikit evidence untuk kelas yang salah.

---

## 6. Gradient Flow: Bagaimana Loss Mengupdate Bobot

```
L_final ──backprop──→ Dempster ──→ α_t ──→ text EDL head ──→ text projection ──→ BERT
                                   α_v ──→ image EDL head ──→ image projection ──→ ResNet
                                   α_c ──→ coattn EDL head ──→ co-attention ──→ kedua branch

L_text ──backprop──→ α_t ──→ text EDL head ──→ text projection ──→ BERT
L_image ──backprop──→ α_v ──→ image EDL head ──→ image projection ──→ ResNet
L_coattn ──backprop──→ α_c ──→ coattn EDL head ──→ co-attention ──→ kedua branch
```

**Multi-path gradient** mencegah overfitting karena:
1. BERT menerima gradient dari L_text + L_coattn + L_final → 3 sumber, tidak terlalu fit ke satu objective
2. Setiap head harus benar secara independen → model tidak bisa "cheat" dengan hanya mengandalkan satu modalitas

---

## 7. Training Strategy Lengkap

```python
# === Hyperparameters ===
backbone_lr = 2e-5          # BERT + ResNet (kecil, avoid overfit)
head_lr = 1e-3              # EDL heads + co-attention (lebih besar)
weight_decay = 0.01         # L2 regularization
annealing_epochs = 10       # KL annealing period
patience = 7                # Early stopping patience
aux_weight = 0.5            # Weight untuk auxiliary heads

# === Optimizer ===
optimizer = AdamW([
    {'params': bert.parameters(), 'lr': backbone_lr},
    {'params': resnet.parameters(), 'lr': backbone_lr},
    {'params': coattn.parameters(), 'lr': head_lr},
    {'params': edl_heads.parameters(), 'lr': head_lr},
], weight_decay=weight_decay)

# === Learning Rate Schedule ===
# Warmup 3 epoch, lalu cosine decay
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

# === Training Loop ===
for epoch in range(max_epochs):
    for batch in train_loader:
        # Forward pass
        alpha_t, alpha_v, alpha_c = model.forward_heads(batch)
        alpha_final = model.dempster_combine(alpha_t, alpha_v, alpha_c)
        
        # Loss computation
        L = aux_weight * head_loss(alpha_t, y, class_weights, epoch) \
          + aux_weight * head_loss(alpha_v, y, class_weights, epoch) \
          + aux_weight * head_loss(alpha_c, y, class_weights, epoch) \
          + 1.0 * head_loss(alpha_final, y, class_weights, epoch)
        
        # Backward + update
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation: monitor Macro-F1 (bukan accuracy!)
    val_f1 = evaluate(model, val_loader)
    early_stopping(val_f1, model)  # save best, stop if no improvement
```

---

## 8. Rangkuman: Mengapa Strategi Ini Mengatasi 3 Masalah Anda

| Masalah | Komponen Loss yang Mengatasi | Mekanisme |
|---------|------------------------------|-----------|
| **Overfitting** | L_var + KL + Annealing + Multi-task | L_var mencegah overconfidence; KL mencegah evidence berlebih; Annealing memberi waktu explore; Multi-task = implicit regularization |
| **Data Imbalance** | Class weights di L_err + KL | Gradient untuk neutral 3.2× lebih kuat; KL juga weighted sehingga misleading evidence terhadap neutral dihukum lebih berat |
| **Update bobot optimal** | L_final + gradient clipping + differential LR | L_final memberikan end-to-end gradient; Gradient clipping mencegah exploding; Backbone LR kecil mencegah catastrophic forgetting |
