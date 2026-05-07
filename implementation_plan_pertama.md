# Arsitektur Model: UA-EDL-CoAttn untuk Multimodal Sentiment Analysis

## 1. Diagnosis Masalah Anda Saat Ini

### Analisis Confusion Matrix

```
                 Predicted
              Neg    Neu    Pos
True Neg     106     21     77     → Recall: 52.0%  (buruk)
True Neu      16     27     27     → Recall: 38.6%  (sangat buruk)
True Pos      74     31    298     → Recall: 73.9%  (lumayan)
```

| Masalah | Penyebab | Dampak |
|---------|----------|--------|
| **Class Imbalance** | Neg:204, Neu:70, Pos:403 → rasio ~3:1:6 | Model bias ke positif |
| **Neutral collapse** | R2 menyebabkan neutral hanya muncul jika KEDUA modalitas neutral → sangat sedikit | Model hampir tidak bisa belajar "neutral" |
| **Neg→Pos confusion** | 77 negative diprediksi positive — model terlalu bias | Kurang discriminative features |
| **Overfitting** | Dataset kecil + model terlalu kompleks tanpa regularisasi yang tepat | Gap train-val besar |

### Mengapa R2 Menyebabkan Masalah Neutral?

```
R2: text=neutral + image=positive → label = POSITIVE (bukan neutral!)
R2: text=positive + image=neutral → label = POSITIVE (bukan neutral!)
```

Akibatnya, label "neutral" hanya tersisa untuk kasus `text=neutral AND image=neutral`. Ini membuat:
1. Jumlah sample neutral **sangat sedikit**
2. Model tidak punya cukup contoh untuk belajar fitur neutral
3. EDL seharusnya bisa membantu: sample neutral = **genuinely uncertain** → uncertainty tinggi

---

## 2. Arsitektur yang Diusulkan: UA-EDL-CoAttn

### Overview Arsitektur (5 Stage)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: (Image, Text)                        │
└────────────────────┬──────────────────────┬─────────────────────────┘
                     │                      │
          ┌──────────▼──────────┐ ┌─────────▼──────────┐
          │  STAGE 1: Feature   │ │  STAGE 1: Feature   │
          │  Extraction (Image) │ │  Extraction (Text)  │
          │  ResNet-50 / ViT    │ │  BERT / RoBERTa     │
          │  → H_v ∈ R^(n×d)   │ │  → H_t ∈ R^(m×d)   │
          └──────────┬──────────┘ └─────────┬──────────┘
                     │                      │
          ┌──────────▼──────────┐ ┌─────────▼──────────┐
          │  STAGE 2: Per-Modal │ │  STAGE 2: Per-Modal │
          │  EDL Head (Image)   │ │  EDL Head (Text)    │
          │  → α_v, u_v         │ │  → α_t, u_t         │
          └──────────┬──────────┘ └─────────┬──────────┘
                     │                      │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │  STAGE 3: Uncertainty│
                     │  -Aware Co-Attention │
                     │  u_v, u_t → modulate │
                     │  cross-attention     │
                     │  → H_v', H_t'        │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │  STAGE 4: Uncertainty│
                     │  -Aware Selection &  │
                     │  Fusion Module       │
                     │  → H_fused           │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │  STAGE 5: Final     │
                     │  EDL Classifier     │
                     │  → α_f, u_f, ŷ      │
                     └─────────────────────┘
```

---

## 3. Detail Setiap Stage

### Stage 1: Feature Extraction

#### Text Branch
```python
# BERT/RoBERTa encoder
H_t = BERT(text_input)          # H_t ∈ R^(m × 768)
H_t = ProjectionLayer(H_t)     # H_t ∈ R^(m × d), d = 256
h_t_cls = H_t[0]               # CLS token representation
```

#### Image Branch
```python
# ResNet-50 (remove FC layer, use feature maps)
H_v = ResNet50(image_input)     # H_v ∈ R^(7×7 × 2048) = R^(49 × 2048)
H_v = ProjectionLayer(H_v)     # H_v ∈ R^(49 × d), d = 256
h_v_avg = AvgPool(H_v)         # Global average pooled representation
```

> [!TIP]
> Gunakan `d = 256` sebagai dimensi proyeksi bersama. Ini cukup ekspresif tapi tidak terlalu besar untuk dataset MVSA yang kecil.

### Stage 2: Per-Modality EDL Head

Setiap modalitas mendapat EDL head sendiri untuk menghasilkan **uncertainty per modalitas**.

```python
# Text EDL
e_t = ReLU(W_t · h_t_cls + b_t)    # evidence text, e_t ∈ R^K (K=3 kelas)
α_t = e_t + 1                       # Dirichlet params
S_t = Σ α_t                         # Dirichlet strength
u_t = K / S_t                       # uncertainty text (scalar)
p̂_t = α_t / S_t                    # predicted probabilities text

# Image EDL
e_v = ReLU(W_v · h_v_avg + b_v)    # evidence image
α_v = e_v + 1
S_v = Σ α_v
u_v = K / S_v                       # uncertainty image (scalar)
p̂_v = α_v / S_v
```

**Mengapa per-modality EDL penting?**
- `u_t` tinggi → teks ambigu/tidak informatif → kurangi pengaruh teks
- `u_v` tinggi → gambar noisy/tidak relevan → kurangi pengaruh gambar
- Untuk kasus R2: jika text=neutral → `u_t` seharusnya tinggi → co-attention otomatis lebih mengandalkan image

### Stage 3: Uncertainty-Aware Co-Attention

Ini adalah **kontribusi utama** — co-attention yang dimodulasi oleh uncertainty.

#### Standard Co-Attention (baseline)
```
# Text-guided visual attention
A_t→v = softmax(H_t · W_a · H_v^T / √d)    # (m × 49)
H_v' = A_t→v^T · H_t                         # (49 × d) attended visual

# Image-guided textual attention  
A_v→t = softmax(H_v · W_b · H_t^T / √d)    # (49 × m)
H_t' = A_v→t^T · H_v                         # (m × d) attended textual
```

#### Uncertainty Modulation (NOVEL)
```
# Confidence scores (inverse of uncertainty)
c_t = 1 - u_t    # confidence text ∈ [0, 1]
c_v = 1 - u_v    # confidence image ∈ [0, 1]

# Temperature scaling berdasarkan uncertainty
τ_t = 1 + β · u_v    # Jika image uncertain, "soften" image→text attention
τ_v = 1 + β · u_t    # Jika text uncertain, "soften" text→image attention

# Uncertainty-modulated co-attention
A_t→v = softmax(H_t · W_a · H_v^T / (√d · τ_v))  × c_v
A_v→t = softmax(H_v · W_b · H_t^T / (√d · τ_t))  × c_t

H_v' = A_t→v^T · H_t    # Attended visual (dimodulasi confidence text)
H_t' = A_v→t^T · H_v    # Attended textual (dimodulasi confidence image)
```

**Intuisi**: Ketika satu modalitas uncertain:
- Temperature naik → attention lebih "spread out" (kurang fokus)
- Confidence rendah → magnitude attention diturunkan
- Efeknya: modalitas yang unreliable punya pengaruh lebih kecil pada modalitas lain

### Stage 4: Uncertainty-Aware Selection & Fusion Module

Modul ini menggabungkan features dari kedua modalitas dengan mempertimbangkan uncertainty.

```python
# Pool attended features
h_t_att = AttentionPool(H_t')    # (d,)
h_v_att = AttentionPool(H_v')    # (d,)

# ===== Uncertainty-Aware Gating =====
# Gate berdasarkan uncertainty
g_t = sigmoid(W_gt · [h_t_att; u_t; u_v])    # gate text ∈ R^d
g_v = sigmoid(W_gv · [h_v_att; u_t; u_v])    # gate image ∈ R^d

# Gated features
h_t_gated = g_t ⊙ h_t_att
h_v_gated = g_v ⊙ h_v_att

# ===== Adaptive Fusion =====
# Fusion weight berdasarkan certainty
w_t = c_t / (c_t + c_v + ε)    # normalized certainty weight text
w_v = c_v / (c_t + c_v + ε)    # normalized certainty weight image

# Weighted fusion
h_fused = w_t · h_t_gated + w_v · h_v_gated

# Residual connection dengan original features
h_final = LayerNorm(h_fused + h_t_cls + h_v_avg)
```

**Mengapa dua level selection?**
1. **Element-wise gating** (`g_t, g_v`): Memilih dimensi fitur mana yang dipercaya — granular
2. **Modality-level weighting** (`w_t, w_v`): Menentukan bobot relatif setiap modalitas — coarse

### Stage 5: Final EDL Classifier + Dempster's Combination

Dua opsi untuk final classifier:

#### Opsi A: EDL Head Langsung (Simple)
```python
e_f = ReLU(MLP(h_final))    # evidence final
α_f = e_f + 1
S_f = Σ α_f
u_f = K / S_f               # final uncertainty
ŷ = α_f / S_f               # final prediction
```

#### Opsi B: Dempster's Combination Rule (Principled) — **DIREKOMENDASIKAN**
```python
# Combine α_t dan α_v menggunakan Dempster's rule (dari TMC paper)
α_combined = DS_Combin(α_t, α_v)

# Optional: combine lagi dengan fused EDL
e_fused = ReLU(MLP(h_final))
α_fused = e_fused + 1
α_final = DS_Combin(α_combined, α_fused)

S_final = Σ α_final
u_final = K / S_final
ŷ = α_final / S_final
```

> [!IMPORTANT]
> **Opsi B** lebih principled secara teoritis karena menggunakan Dempster's rule untuk mengombinasikan evidence dari berbagai sumber. Ini juga memberikan **3 level uncertainty**: per-modalitas + combined + fused.

---

## 4. Loss Function: Multi-Task EDL Loss

### Total Loss
```
L_total = λ₁·L_EDL_text + λ₂·L_EDL_image + λ₃·L_EDL_final + λ₄·L_class_balance
```

### Per-component EDL Loss (menggunakan SSE Bayes Risk + KL)
```
L_EDL(α, y) = Σⱼ [(yⱼ - α_j/S)² + α_j(S-α_j)/(S²(S+1))]     ← SSE Bayes Risk
            + λ_t · KL[Dir(α̃) ‖ Dir(1)]                         ← KL Regularizer
```

### Class-Balance Loss (untuk masalah neutral)
```python
# Class weights (inverse frequency)
class_counts = [N_neg, N_neu, N_pos]
class_weights = 1.0 / torch.tensor(class_counts)
class_weights = class_weights / class_weights.sum() * K    # normalize

# Weighted EDL loss
L_EDL_weighted = Σᵢ class_weights[yᵢ] · L_EDL(αᵢ, yᵢ)
```

### Recommended λ values
```
λ₁ = 1.0    (text EDL)
λ₂ = 1.0    (image EDL)  
λ₃ = 1.0    (final EDL)
λ₄ = 0.5    (class balance — tunable)
```

> [!TIP]
> Multi-task loss (text + image + final) berfungsi sebagai **regularizer implicit** karena setiap branch harus menghasilkan prediksi yang masuk akal. Ini membantu mengatasi overfitting.

---

## 5. Strategi Mengatasi Masalah Spesifik

### 5.1 Mengatasi Overfitting

| Strategi | Implementasi |
|----------|-------------|
| **EDL KL Regularizer** | Sudah built-in di loss → mencegah overconfident predictions |
| **Multi-task loss** | 3 supervision signals → regularisasi implisit |
| **Dropout** | 0.3-0.5 pada projection layers dan MLP |
| **Freeze backbone** | Freeze BERT & ResNet di 5 epoch pertama, lalu fine-tune dengan lr kecil |
| **Early stopping** | Berdasarkan validation macro-F1 (bukan accuracy!) |
| **Image augmentation** | RandomResizedCrop, HorizontalFlip, ColorJitter |
| **Label smoothing** | Ringan (0.1) pada loss tambahan selain EDL |

### 5.2 Mengatasi Neutral Class Problem

| Strategi | Mekanisme |
|----------|-----------|
| **Class-weighted loss** | Weight neutral 3-6× lebih besar |
| **EDL uncertainty** | Sample neutral memiliki uncertainty natural tinggi → model belajar bahwa "uncertain = possibly neutral" |
| **Uncertainty threshold** | Jika `u_final > threshold` → pertimbangkan prediksi sebagai neutral |
| **Oversampling neutral** | Duplicate neutral samples 2-3× di training set |
| **Focal-EDL hybrid** | Kurangi loss untuk easy samples, fokus pada hard neutral samples |

### 5.3 Uncertainty-Based Neutral Detection (Novel)

```python
# Post-processing rule menggunakan uncertainty
def predict_with_uncertainty(α_final, u_final, threshold=0.6):
    pred_class = argmax(α_final / S_final)
    
    # Jika uncertainty tinggi dan predicted bukan neutral → reconsider
    if u_final > threshold:
        return "neutral"  # Model tidak yakin → default ke neutral
    else:
        return pred_class
```

> [!IMPORTANT]
> Ini adalah insight kunci: pada dataset MVSA dengan preprocessing R2, sample yang **genuinely neutral** memang ambigu. EDL bisa menangkap ambiguitas ini sebagai **high uncertainty**, yang kemudian bisa digunakan sebagai sinyal tambahan bahwa sample tersebut kemungkinan neutral.

---

## 6. Hyperparameter yang Direkomendasikan

| Parameter | Value | Catatan |
|-----------|-------|---------|
| Projection dim `d` | 256 | Balance antara capacity dan overfitting |
| Learning rate (backbone) | 1e-5 | Fine-tune BERT/ResNet |
| Learning rate (heads) | 1e-3 | EDL heads + attention + fusion |
| Batch size | 32 | Disesuaikan GPU memory |
| Epochs | 30-50 | Dengan early stopping patience=7 |
| KL annealing steps | 10 epoch | Sesuai paper EDL |
| Uncertainty modulation β | 1.0 | Temperature scaling factor |
| Dropout | 0.3 | Semua projection/MLP layers |
| Optimizer | AdamW | Weight decay = 0.01 |

---

## 7. Evaluation Metrics

```
Primary:   Macro-F1 Score (treats all classes equally)
Secondary: Weighted-F1, Per-class F1, Accuracy
Analysis:  
  - Uncertainty distribution per class
  - Uncertainty vs correctness correlation
  - Per-modality uncertainty analysis
  - Confusion matrix with uncertainty overlay
```

---

## 8. Novelty / Kontribusi Tesis

| # | Kontribusi | Deskripsi |
|---|-----------|-----------|
| 1 | **Per-modality EDL** | EDL head terpisah untuk setiap modalitas → uncertainty per modalitas |
| 2 | **Uncertainty-modulated Co-Attention** | Temperature scaling & confidence gating pada cross-attention berdasarkan uncertainty |
| 3 | **Uncertainty-Aware Selection Module** | Dual-level selection (element-wise gate + modality-level weight) berdasarkan EDL uncertainty |
| 4 | **Multi-level evidence fusion** | Dempster's rule untuk combine evidence dari 3 sumber (text, image, fused) |
| 5 | **Uncertainty-based neutral detection** | Memanfaatkan high uncertainty sebagai sinyal neutral class |

---

## Open Questions

> [!IMPORTANT]
> 1. **Backbone**: Apakah Anda prefer BERT atau RoBERTa untuk teks? ResNet-50 atau ViT untuk gambar?
> 2. **Final classifier**: Prefer Opsi A (simple EDL head) atau Opsi B (Dempster's Combination)?
> 3. **Dataset split**: Berapa rasio train/val/test yang Anda gunakan?
> 4. **Baseline**: Model apa yang menjadi baseline perbandingan Anda?
> 5. **Kode**: Apakah Anda ingin saya implementasikan arsitektur ini dalam PyTorch?
