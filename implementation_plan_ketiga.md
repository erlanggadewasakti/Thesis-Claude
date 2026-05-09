# Plan Perbaikan Komprehensif: UA-EDL-CoAttn v2 → v3
**Target: Macro-F1 > 80% pada MVSA Dataset**

---

## 1. Diagnosis Mendalam — Mengapa v2 Hanya Mencapai 61%?

### 1.1 Ringkasan Hasil v2

```
Best Val Macro-F1: 0.6533 (epoch 25)
Test Macro-F1:     0.6119
Test Accuracy:     0.7312

Per-class:           Precision  Recall  F1
  negative (n=204):    0.6927   0.6078  0.6475
  neutral  (n=70):     0.6333   0.2714  0.3800  ← bottleneck
  positive (n=403):    0.7521   0.8734  0.8083

Overfitting gap:  Train F1=0.92, Val F1=0.65 → gap=0.27
```

### 1.2 Empat Root Causes

---

#### 🔴 Root Cause 1: Class Weight = FLAT (Paling Kritis)

**Bukti:**
```
Class weights tensor: [0.9875, 1.0250, 0.9875]
                       neg     neu     pos
```

**Mengapa ini terjadi?** Effective Number dengan `β=0.99`:
```python
# Train distribution: neg=950, neu=329, pos=1878
effective_num(950)  = 1 - 0.99^950  = 1 - 7.2e-5  ≈ 1.0000
effective_num(329)  = 1 - 0.99^329  = 1 - 0.0369   ≈ 0.9631
effective_num(1878) = 1 - 0.99^1878 = 1 - 5.5e-9   ≈ 1.0000

weights_raw = 0.01 / [1.0000, 0.9631, 1.0000] = [0.01, 0.01038, 0.01]
normalized ×3 = [0.9875, 1.0250, 0.9875]
```

**Dampak:** Neutral diberi bobot 1.025 — **praktis tidak ada perbedaan** dengan kelas lain. Model tidak mendapat insentif tambahan untuk memprediksi neutral dengan benar.

**Fix:** Ganti ke inverse sqrt frequency → neutral weight ≈ 1.50 (2.4× lebih besar dari positive).

---

#### 🔴 Root Cause 2: Image Head MATI (Tidak Belajar)

**Bukti — Uncertainty per head:**
```
Head          neg-u    neu-u    pos-u    Interpretasi
Text (u_t):   0.640    0.717    0.485    Lumayan (u < 0.7 untuk neg/pos)
Image (u_v):  0.866    0.899    0.753    MATI (u ≈ 0.9 = total ignorance)
CoAttn (u_c): 0.525    0.615    0.379    Baik (paling informatif)
Final (u_f):  0.340    0.464    0.199    OK (Dempster combine berhasil)
```

**Mengapa image head mati?**
1. ResNet-50 dilatih untuk object recognition (ImageNet), bukan visual sentiment
2. Mean pooling atas 49 region → informasi spasial hilang
3. Sentiment visual seringkali terletak di **area kecil** (ekspresi wajah, warna dominan) yang di-average-out oleh mean pool

**Dampak:** Dempster's combination hanya efektif menggabungkan text + co-attention. Image head hampir tidak berkontribusi — seolah model hanya 2-head.

**Fix:** Ganti mean pooling → Attention Pooling (learned, weighted pooling yang fokus pada region informatif).

---

#### 🟡 Root Cause 3: Overfitting Masih Ada (Gap 0.27)

**Bukti — Training dynamics:**
```
Epoch  8 (frozen): Train L=1.03, Val L=1.07 → gap=0.04 ✅
Epoch 12 (unfroze): Train L=0.87, Val L=0.96 → gap=0.09 ⚠️
Epoch 19:          Train L=0.44, Val L=1.22 → gap=0.78 ❌
Epoch 25 (best):   Train L=0.29, Val L=1.39 → gap=1.10 ❌
Epoch 35 (stop):   Train L=0.17, Val L=1.43 → gap=1.26 ❌❌
```

**Pattern:** Val loss mulai naik setelah epoch ~12 (4 epoch setelah unfreeze). Model menghafal training data setelah backbone di-unfreeze.

**Fix:** Mixup augmentation + lebih banyak regularisasi pada backbone.

---

#### 🟠 Root Cause 4: Neutral Selalu Diprediksi Positive

**Bukti — Confusion matrix test:**
```
                  Pred_neg  Pred_neu  Pred_pos
True neutral:        7        19        46     ← 66% neutral → positive!
```

**Mengapa?** Kombinasi:
1. Class weight flat (1.025) → loss tidak lebih berat untuk neutral
2. Positive dominan di training (1878/3157 = 59%) → model defaultnya prediksi positive
3. Neutral sering mirip positive secara fitur (teks netral + gambar biasa)

**Fix:** WeightedRandomSampler + neutral boost di loss.

---

## 2. Rencana Perbaikan — 6 Perubahan

### Overview Perubahan

```
┌─────────────────────────────────────────────────────────────┐
│              PERUBAHAN v2 → v3                              │
├──────────┬──────────────────────────┬───────────┬───────────┤
│ File     │ Perubahan                │ Priority  │ Impact    │
├──────────┼──────────────────────────┼───────────┼───────────┤
│ 04       │ Fix class weight         │ 🔴 P0     │ +5-8%    │
│ 04       │ WeightedRandomSampler    │ 🔴 P0     │ +3-5%    │
│ 05       │ AttentionPool image      │ 🟠 P1     │ +2-4%    │
│ 06       │ Neutral boost in loss    │ 🔴 P0     │ +3-5%    │
│ 07       │ Mixup augmentation       │ 🟠 P1     │ +3-5%    │
│ 08       │ Layerwise LR BERT        │ 🟡 P2     │ +1-2%    │
└──────────┴──────────────────────────┴───────────┴───────────┘
Total estimated gain: +17-29% → target 0.61 + 0.20 = 0.81
```

---

### Perbaikan 1: Fix Class Weight — Inverse Sqrt Frequency 🔴

#### [MODIFY] [04_dataset.py](file:///e:/Thesis%20Claude/cells/04_dataset.py)

**Sebelum (line 39-47):**
```python
class_counts = train_df["label"].value_counts().sort_index().values.astype(float)
beta = HP.CLASS_WEIGHT_BETA
effective_num = 1.0 - np.power(beta, class_counts)
weights = (1.0 - beta) / effective_num
class_weights = torch.tensor(
    weights / weights.sum() * HP.NUM_CLASSES, dtype=torch.float32
).to(device)
```

**Sesudah:**
```python
class_counts = train_df["label"].value_counts().sort_index().values.astype(float)
weights = 1.0 / np.sqrt(class_counts)
class_weights = torch.tensor(
    weights / weights.sum() * HP.NUM_CLASSES, dtype=torch.float32
).to(device)
```

**Perbandingan weight:**
```
                    Effective Number (v2)    Inverse Sqrt (v3)
negative (950):         0.9875                   0.878
neutral  (329):         1.0250                   1.496  ← 46% lebih besar!
positive (1878):        0.9875                   0.626  ← 37% lebih kecil!

Rasio neu/pos:          1.04×                    2.39×  ← jauh lebih diskriminatif
```

Juga hapus `CLASS_WEIGHT_BETA` dari HP karena tidak dipakai lagi.

---

### Perbaikan 2: WeightedRandomSampler untuk Oversample Neutral 🔴

#### [MODIFY] [04_dataset.py](file:///e:/Thesis%20Claude/cells/04_dataset.py)

**Sesudah blok DataLoader creation (line 108-118), ganti menjadi:**
```python
from torch.utils.data import WeightedRandomSampler

# Per-sample weights: oversample minority classes
label_to_weight = {
    0: 1878 / 950,    # neg: ~2.0×
    1: 1878 / 329,    # neu: ~5.7×
    2: 1.0,           # pos: baseline
}
sample_weights = [label_to_weight[l] for l in train_df["label"].values]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=HP.BATCH_SIZE,
    sampler=sampler,          # ganti shuffle=True
    num_workers=2, pin_memory=True, drop_last=False
)
```

**Efek:** Setiap epoch, model melihat distribusi yang hampir seimbang:
```
Sebelum: neg=950(30%), neu=329(10%), pos=1878(60%)
Sesudah (expected per epoch): neg≈1200(38%), neu≈1200(38%), pos≈757(24%)
```

---

### Perbaikan 3: AttentionPool untuk Image Head 🟠

#### [MODIFY] [05_model.py](file:///e:/Thesis%20Claude/cells/05_model.py)

**Tambah class AttentionPool sebelum class UAEDLCoAttn:**
```python
class AttentionPool(nn.Module):
    """Learned attention pooling — focuses on informative regions."""
    def __init__(self, dim):
        super().__init__()
        self.attn_w = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, N, d)
        scores = self.attn_w(x)           # (B, N, 1)
        weights = F.softmax(scores, dim=1) # (B, N, 1)
        return (weights * x).sum(dim=1)    # (B, d)
```

**Dalam UAEDLCoAttn.__init__, tambah:**
```python
self.image_attn_pool = AttentionPool(proj_dim)
self.text_attn_pool = AttentionPool(proj_dim)
```

**Dalam forward, ganti pooling:**
```python
# Sebelum:
h_v_pool = H_v.mean(dim=1)
h_t_cls = H_t[:, 0, :]

# Sesudah:
h_v_pool = self.image_attn_pool(H_v)     # learned attention over 49 regions
h_t_cls = self.text_attn_pool(H_t)       # learned attention over tokens

# Co-attention head juga pakai attention pool:
h_v_att = self.image_attn_pool(H_v_prime)
h_t_att = self.text_attn_pool(H_t_prime)
```

**Mengapa ini membantu image head:**
- Mean pool: semua 49 region diberi bobot sama → informasi sentiment di-dilute
- Attention pool: model belajar region mana yang paling informatif (wajah, ekspresi, warna) → fitur visual lebih ekspresif → evidence lebih bermakna

---

### Perbaikan 4: Neutral-Aware Boost di Loss 🔴

#### [MODIFY] [06_loss.py](file:///e:/Thesis%20Claude/cells/06_loss.py)

**Modifikasi focal_edl_loss (line 5-25):**
```python
def focal_edl_loss(alpha, y_onehot, class_weights, gamma=2.0):
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S

    p_correct = (p_hat * y_onehot).sum(dim=1, keepdim=True)
    focal_w = (1 - p_correct.detach()) ** gamma

    err = (y_onehot - p_hat) ** 2
    var = p_hat * (1 - p_hat) / (S + 1)

    # Neutral-aware boost: extra 2× penalty for neutral samples
    # Ini STACKS dengan class_weight — neutral total mendapat ~3× boost
    neutral_mask = y_onehot[:, 1:2]               # (B, 1) — 1 jika neutral
    sample_boost = 1.0 + 2.0 * neutral_mask       # (B, 1) — 3× untuk neutral, 1× lainnya

    loss = focal_w * sample_boost * class_weights.unsqueeze(0) * (err + var)
    return loss.sum(dim=1).mean()
```

**Total efektif weight untuk neutral:**
```
class_weight(neu) × sample_boost(neu) = 1.496 × 3.0 = 4.49
vs class_weight(pos) × sample_boost(pos) = 0.626 × 1.0 = 0.626

→ Neutral loss 7.2× lebih berat dari positive per sample!
```

Ini agresif tapi diperlukan mengingat neutral hanya 10% dari data dan recall saat ini 27%.

---

### Perbaikan 5: Mixup Data Augmentation 🟠

#### [MODIFY] [07_train_eval.py](file:///e:/Thesis%20Claude/cells/07_train_eval.py)

**Tambah fungsi mixup sebelum train_one_epoch:**
```python
def mixup_batch(images, y_onehot, alpha=0.4):
    """
    Mixup: interpolasi antar-sampel untuk augmentasi.
    Hanya untuk image — text tidak bisa di-mix (discrete tokens).
    """
    if alpha <= 0:
        return images, y_onehot

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Pastikan lam >= 0.5 (sample asli dominan)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]

    return mixed_images, mixed_y
```

**Modifikasi train_one_epoch — setelah membuat y_onehot, tambah:**
```python
# Mixup (50% chance per batch, hanya setelah backbone unfreeze)
use_mixup = (epoch >= HP.BACKBONE_FREEZE_EPOCHS) and (random.random() < 0.5)
if use_mixup:
    images, y_onehot = mixup_batch(images, y_onehot, alpha=0.4)
```

**Mengapa mixup membantu:**
1. **Anti-overfit:** Memperluas effective dataset size tanpa data baru
2. **Smoother decision boundary:** Model tidak overfit ke individual samples
3. **Implicit regularization:** Setara dengan menambah noise ke training → generalisasi lebih baik

---

### Perbaikan 6: Layerwise LR Decay untuk BERT 🟡

#### [MODIFY] [08_training.py](file:///e:/Thesis%20Claude/cells/08_training.py)

**Ganti optimizer setup (line 5-15):**
```python
def get_layerwise_lr_params(model, bert_base_lr=5e-6, lr_decay=0.85, head_lr=5e-4):
    """BERT layer-wise LR decay: lower layers get smaller LR."""
    params = []

    # BERT embeddings — smallest LR
    params.append({
        "params": model.text_backbone.embeddings.parameters(),
        "lr": bert_base_lr * (lr_decay ** 12)
    })

    # BERT encoder layers — gradual increase
    for i, layer in enumerate(model.text_backbone.encoder.layer):
        lr = bert_base_lr * (lr_decay ** (11 - i))
        params.append({"params": layer.parameters(), "lr": lr})

    # BERT pooler
    if model.text_backbone.pooler is not None:
        params.append({
            "params": model.text_backbone.pooler.parameters(),
            "lr": bert_base_lr
        })

    # ResNet — same backbone LR
    params.append({"params": model.image_backbone.parameters(), "lr": bert_base_lr})

    # All heads — higher LR
    for module in [model.text_proj, model.image_proj, model.co_attention,
                   model.text_edl_head, model.image_edl_head, model.coattn_edl_head,
                   model.image_attn_pool, model.text_attn_pool]:
        params.append({"params": module.parameters(), "lr": head_lr})

    return params

optimizer = torch.optim.AdamW(
    get_layerwise_lr_params(model),
    weight_decay=HP.WEIGHT_DECAY
)
```

**LR distribution:**
```
BERT Layer  0: 5e-6 × 0.85^11 = 8.6e-7  (hampir frozen)
BERT Layer  5: 5e-6 × 0.85^6  = 1.9e-6
BERT Layer 11: 5e-6 × 0.85^0  = 5.0e-6  (full LR)
Heads:                           5.0e-4  (100× BERT top)
```

**Mengapa:** Layer awal BERT menangkap fitur umum (syntax, morphology) yang tidak perlu banyak berubah. Layer akhir menangkap fitur semantik yang perlu di-tune untuk sentiment.

---

## 3. Perubahan HP yang Juga Diperlukan

#### [MODIFY] [04_dataset.py](file:///e:/Thesis%20Claude/cells/04_dataset.py) — HP class

```python
class HP:
    PROJ_DIM = 128
    BACKBONE_LR = 5e-6
    HEAD_LR = 5e-4
    LR_DECAY = 0.85          # NEW: layerwise LR decay factor
    WEIGHT_DECAY = 0.02
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    KL_ANNEALING_EPOCHS = 5
    FOCAL_GAMMA = 2.0
    # CLASS_WEIGHT_BETA dihapus — tidak dipakai lagi
    DROPOUT = 0.5
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 10
    BACKBONE_FREEZE_EPOCHS = 8
    MAX_TEXT_LEN = 128
    NUM_CLASSES = 3
    IMG_SIZE = 224
    MIXUP_ALPHA = 0.4         # NEW: mixup interpolation strength
    NEUTRAL_BOOST = 2.0       # NEW: extra penalty multiplier for neutral
```

---

## 4. Verification Plan

### 4.1 Metrik yang Dimonitor

| Metrik | Target v3 | v2 Saat Ini | Cara Verifikasi |
|--------|-----------|-------------|-----------------|
| Test Macro-F1 | **> 0.80** | 0.6119 | Classification report |
| Neutral Recall | **> 0.55** | 0.2714 | Per-class metrics |
| Neutral F1 | **> 0.50** | 0.3800 | Per-class metrics |
| Overfitting gap | **< 0.15** | 0.27 | Train F1 - Val F1 |
| Image u_v | **< 0.70** | 0.87 | Uncertainty stats |
| Correct u_f mean | **< 0.20** | 0.2206 | Uncertainty analysis |

### 4.2 Automated Tests

```bash
# Setelah training, run cell 10 (test eval) dan cell 11 (uncertainty)
# Cek:
# 1. classification_report → neutral recall > 0.55
# 2. confusion matrix → neutral→positive count < 30 (saat ini 46)
# 3. uncertainty stats → u_v < 0.70 (saat ini 0.87)
# 4. training curve → val loss gap < train loss × 2
```

### 4.3 Ablation Study (Opsional, untuk Tesis)

Jika semua fix diterapkan sekaligus dan mencapai target, lakukan ablation untuk mengukur kontribusi masing-masing:

| Eksperimen | Konfigurasi |
|------------|-------------|
| v3 Full | Semua 6 perbaikan |
| v3 − class weight | Kembali ke effective number |
| v3 − sampler | Kembali ke shuffle (tanpa oversample) |
| v3 − attention pool | Kembali ke mean pool |
| v3 − mixup | Tanpa mixup |
| v3 − neutral boost | Tanpa neutral boost di loss |
| v3 − layerwise LR | Kembali ke flat LR |

---

## 5. Ringkasan Flow Perubahan

```
v2 (saat ini):
  class_weight ≈ flat (1.025) → model abaikan neutral
  mean_pool(49 regions) → image head mati (u=0.87)
  no oversampling → neutral underrepresented (10%)
  no mixup → overfit (gap 0.27)

v3 (target):
  class_weight = [0.88, 1.50, 0.63] → neutral 2.4× pos ✅
  WeightedRandomSampler → neutral 38% per epoch ✅
  AttentionPool → image head fokus pada region informatif ✅
  neutral_boost 3× di loss → 7.2× total vs positive ✅
  mixup 50% → smoother boundaries, less overfit ✅
  layerwise LR → BERT fine-tune lebih halus ✅
```

## Open Questions

> [!IMPORTANT]
> 1. Apakah Anda setuju dengan semua 6 perbaikan? Atau ada yang ingin diubah/dihapus?
> 2. Apakah Anda ingin saya langsung implementasikan semua sekaligus, atau bertahap untuk melihat dampak per perbaikan?
> 3. Neutral boost = 2.0 (total efektif 7.2× vs positive) — apakah ini terlalu agresif menurut Anda, atau bisa dicoba dulu?
