# Rancangan Final: UA-EDL-CoAttn v2
**Integrasi Modul Uncertainty-Aware Selection dengan Evidential Deep Learning pada Mekanisme Co-Attention untuk Analisis Sentimen Multimodal**

Dataset: MVSA-Single & MVSA-Multiple | Kelas: Negative, Neutral, Positive

---

## 1. Overview Arsitektur

```
┌──────────────────────────────────────────────────────────────────────┐
│                       INPUT: (Image, Text)                          │
└───────────────┬──────────────────────────────┬───────────────────────┘
                │                              │
     ┌──────────▼──────────┐        ┌──────────▼──────────┐
     │  STAGE 1a: Image    │        │  STAGE 1b: Text     │
     │  ResNet-50 (frozen  │        │  BERT (frozen awal) │
     │  awal, fine-tune)   │        │                     │
     │  → H_v ∈ R^(49×d)  │        │  → H_t ∈ R^(m×d)   │
     └──────┬──────────────┘        └──────────┬──────────┘
            │         ┌────────────────────┐   │
            │    ┌────┤  STAGE 2:          ├───┤
            │    │    │  Co-Attention       │   │
            │    │    │  (Standard, tanpa   │   │
            │    │    │   uncertainty)      │   │
            │    │    │  → H_v', H_t'      │   │
            │    │    └────────┬────────────┘   │
            │    │             │                │
     ┌──────▼────┤    ┌───────▼────────┐  ┌────▼──────┐
     │ STAGE 3a  │    │ STAGE 3b       │  │ STAGE 3c  │
     │ HEAD      │    │ HEAD           │  │ HEAD      │
     │ IMAGE     │    │ CO-ATTENTION   │  │ TEXT      │
     │           │    │                │  │           │
     │ AvgPool   │    │ Pool(H_v')     │  │ CLS token │
     │ (H_v)     │    │ ⊕ Pool(H_t')  │  │ (H_t)     │
     │ → MLP     │    │ → MLP          │  │ → MLP     │
     │ → ReLU    │    │ → ReLU         │  │ → ReLU    │
     │ → e_v     │    │ → e_c          │  │ → e_t     │
     │ α_v=e+1   │    │ α_c=e+1        │  │ α_t=e+1   │
     │ u_v=K/S_v │    │ u_c=K/S_c      │  │ u_t=K/S_t │
     └─────┬─────┘    └───────┬────────┘  └─────┬─────┘
           │     L_image      │  L_coattn        │  L_text
           │                  │                  │
           │    ┌─────────────▼──────────────┐   │
           └───►│  STAGE 4:                  │◄──┘
                │  Dempster's Combination    │
                │  α_f = DS(α_t, α_v, α_c)  │
                └────────────┬───────────────┘
                             │  L_final
                   ┌─────────▼───────────┐
                   │  STAGE 5: Output    │
                   │  ŷ = α_f / S_f      │
                   │  u = K / S_f        │
                   └─────────────────────┘
```

### Alur Data Ringkas

| Stage | Input | Output | Fungsi |
|-------|-------|--------|--------|
| 1a, 1b | Raw image, raw text | `H_v (49×d)`, `H_t (m×d)` | Feature extraction + projection ke dimensi bersama |
| 2 | `H_v`, `H_t` | `H_v'`, `H_t'` | Cross-modal interaction via co-attention |
| 3a | `H_v` (sebelum co-attn) | `α_v, u_v` | Unimodal image evidence + uncertainty |
| 3b | `H_v', H_t'` (setelah co-attn) | `α_c, u_c` | Cross-modal evidence + uncertainty |
| 3c | `H_t` (sebelum co-attn) | `α_t, u_t` | Unimodal text evidence + uncertainty |
| 4 | `α_t, α_v, α_c` | `α_f` | Dempster's combination dari 3 sumber |
| 5 | `α_f` | `ŷ, u_f` | Prediksi final + overall uncertainty |

---

## 2. Detail Formulasi Setiap Stage

### Stage 1a: Image Feature Extraction

```python
# ResNet-50, hapus FC layer, ambil feature maps dari layer4
H_v_raw = ResNet50(image)                    # (batch, 2048, 7, 7)
H_v_flat = H_v_raw.flatten(2).permute(0,2,1) # (batch, 49, 2048)
H_v = Dropout(Linear(H_v_flat, d))           # (batch, 49, d)  d=256
```

### Stage 1b: Text Feature Extraction

```python
# BERT encoder, ambil semua token representations
H_t_raw = BERT(text_input)                   # (batch, m, 768)
H_t = Dropout(Linear(H_t_raw, d))            # (batch, m, d)   d=256
```

### Stage 2: Co-Attention (Standard Bidirectional)

```python
# Text-guided visual attention
# Query: text, Key/Value: image
A_t2v = softmax( (H_t @ W_a @ H_v.T) / √d )  # (batch, m, 49)
H_v_prime = A_t2v.transpose(-1,-2) @ H_t       # (batch, 49, d)

# Image-guided textual attention
# Query: image, Key/Value: text
A_v2t = softmax( (H_v @ W_b @ H_t.T) / √d )  # (batch, 49, m)
H_t_prime = A_v2t.transpose(-1,-2) @ H_v       # (batch, m, d)
```

Co-attention di sini **tidak dimodulasi uncertainty** — ini by design. Uncertainty dihitung SETELAH co-attention bekerja, bukan sebelumnya. Ini menghindari masalah "premature uncertainty" dari rancangan pertama.

### Stage 3a: Image EDL Head

```python
# Input: H_v (features SEBELUM co-attention) → genuine unimodal
h_v_pool = H_v.mean(dim=1)                     # (batch, d) — average pooling
h_v_mlp = Dropout(ReLU(Linear(h_v_pool, d)))    # (batch, d)
e_v = ReLU(Linear(h_v_mlp, K))                  # (batch, K) — evidence, K=3

α_v = e_v + 1                                    # (batch, K) — Dirichlet params
S_v = α_v.sum(dim=1, keepdim=True)               # (batch, 1) — Dirichlet strength
u_v = K / S_v                                    # (batch, 1) — uncertainty
```

### Stage 3b: Co-Attention EDL Head

```python
# Input: H_v', H_t' (features SETELAH co-attention) → cross-modal
h_v_att = H_v_prime.mean(dim=1)                  # (batch, d)
h_t_att = H_t_prime.mean(dim=1)                  # (batch, d)
h_cross = torch.cat([h_v_att, h_t_att], dim=-1)  # (batch, 2d)

h_cross_mlp = Dropout(ReLU(Linear(h_cross, d)))   # (batch, d)
e_c = ReLU(Linear(h_cross_mlp, K))                # (batch, K) — evidence

α_c = e_c + 1
S_c = α_c.sum(dim=1, keepdim=True)
u_c = K / S_c
```

### Stage 3c: Text EDL Head

```python
# Input: H_t (features SEBELUM co-attention) → genuine unimodal
h_t_cls = H_t[:, 0, :]                           # (batch, d) — CLS token
h_t_mlp = Dropout(ReLU(Linear(h_t_cls, d)))       # (batch, d)
e_t = ReLU(Linear(h_t_mlp, K))                    # (batch, K) — evidence

α_t = e_t + 1
S_t = α_t.sum(dim=1, keepdim=True)
u_t = K / S_t
```

### Stage 4: Dempster's Combination Rule

```python
def DS_Combin_two(alpha1, alpha2, K):
    """Combine 2 sumber evidence via Dempster's rule."""
    # Belief & uncertainty dari masing-masing sumber
    S1, S2 = alpha1.sum(1, keepdim=True), alpha2.sum(1, keepdim=True)
    E1, E2 = alpha1 - 1, alpha2 - 1
    b1, b2 = E1 / S1, E2 / S2
    u1, u2 = K / S1, K / S2

    # Conflict coefficient
    bb = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))      # (batch, K, K)
    C = bb.sum(dim=(1,2)) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)

    # Combined belief & uncertainty
    b_comb = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C).unsqueeze(1)
    u_comb = (u1 * u2) / (1 - C).unsqueeze(1)

    # Convert back to Dirichlet params
    S_comb = K / u_comb
    e_comb = b_comb * S_comb
    alpha_comb = e_comb + 1
    return alpha_comb

# Combine 3 sources: unimodal dulu, lalu cross-modal
α_tv = DS_Combin_two(α_t, α_v, K)      # text + image
α_f  = DS_Combin_two(α_tv, α_c, K)     # (text+image) + co-attention
```

**Mengapa urutan ini?** Dua sumber unimodal (text, image) dikombinasi duluan → menghasilkan fused unimodal evidence. Lalu dikombinasikan dengan co-attention evidence yang menangkap interaksi cross-modal. Ini memastikan unimodal dan cross-modal evidence punya peran yang seimbang.

### Stage 5: Output

```python
S_f = α_f.sum(dim=1, keepdim=True)
p_hat = α_f / S_f                # predicted probabilities
u_f = K / S_f                    # overall uncertainty

prediction = p_hat.argmax(dim=1)  # predicted class
```

---

## 3. Loss Function

### 3.1 Total Loss

```
L_total = w_aux(t) × (L_text + L_image + L_coattn) + 1.0 × L_final
```

di mana `w_aux(t) = max(0.1, 1.0 − t/T)` menurun dari 1.0 ke 0.1 seiring training.

**Mengapa dynamic auxiliary weight?**
- Awal training (w_aux=1.0): setiap head dipaksa belajar mandiri → regularisasi kuat
- Akhir training (w_aux=0.1): L_final mendominasi → fine-tune kolaborasi via Dempster

### 3.2 Per-Head Loss: Focal-EDL + KL Regularizer

Setiap `L_head` terdiri dari:

```
L_head = L_focal_edl(α, y) + λ_t × L_kl(α, y)
```

#### Komponen 1: Focal-EDL Loss (SSE Bayes Risk + Effective Number Weight + Focal Weight)

```python
def focal_edl_loss(alpha, y_onehot, class_weights, gamma=1.0):
    S = alpha.sum(dim=1, keepdim=True)            # (batch, 1)
    p_hat = alpha / S                              # (batch, K)

    # --- Focal weight (per sample, dynamic) ---
    p_correct = (p_hat * y_onehot).sum(dim=1, keepdim=True)  # (batch, 1)
    focal_w = (1 - p_correct) ** gamma             # (batch, 1)

    # --- EDL loss components ---
    err = (y_onehot - p_hat) ** 2                  # (batch, K)
    var = p_hat * (1 - p_hat) / (S + 1)            # (batch, K)

    # Class weight (static) × focal weight (dynamic)
    loss = focal_w * class_weights * (err + var)    # (batch, K)

    return loss.sum(dim=1).mean()
```

#### Komponen 2: KL Divergence Regularizer

```python
def kl_divergence(alpha, y_onehot, epoch, annealing_epochs=10):
    lambda_t = min(1.0, epoch / annealing_epochs)

    # Remove evidence kelas benar
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha    # kelas benar → 1

    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

    # KL[Dir(α̃) || Dir(1)]
    K = alpha.shape[1]
    ln_B = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(1, keepdim=True)
    ln_B_uni = torch.lgamma(torch.ones(1, K)).sum() - torch.lgamma(torch.tensor(K * 1.0))

    dg0 = torch.digamma(S_tilde)
    dg1 = torch.digamma(alpha_tilde)

    kl = ((alpha_tilde - 1) * (dg1 - dg0)).sum(1, keepdim=True) + ln_B + ln_B_uni

    return lambda_t * kl.mean()
```

### 3.3 Class Weight: Effective Number of Samples

```python
beta = 0.99
class_counts = torch.tensor([N_neg, N_neu, N_pos])   # e.g. [204, 70, 403]
effective_num = 1.0 - beta ** class_counts
weights = (1.0 - beta) / effective_num
class_weights = weights / weights.sum() * K            # normalize, mean ≈ 1
```

### 3.4 Full Loss Computation

```python
def total_loss(α_t, α_v, α_c, α_f, y, cw, epoch, max_epoch, gamma=1.0):
    L_t = focal_edl_loss(α_t, y, cw, gamma) + kl_divergence(α_t, y, epoch)
    L_v = focal_edl_loss(α_v, y, cw, gamma) + kl_divergence(α_v, y, epoch)
    L_c = focal_edl_loss(α_c, y, cw, gamma) + kl_divergence(α_c, y, epoch)
    L_f = focal_edl_loss(α_f, y, cw, gamma) + kl_divergence(α_f, y, epoch)

    w_aux = max(0.1, 1.0 - epoch / max_epoch)

    return w_aux * (L_t + L_v + L_c) + 1.0 * L_f
```

### Ringkasan Peran Setiap Komponen Loss

| Komponen | Mengatasi | Cara Kerja |
|----------|-----------|------------|
| **L_err** `(y−p̂)²` | Prediksi salah | Gradient besar jika prediksi jauh dari label |
| **L_var** `p̂(1−p̂)/(S+1)` | Overconfidence | Model tidak boleh yakin tanpa bukti cukup |
| **Focal weight** `(1−p_correct)^γ` | Overfit ke easy samples | Model fokus pada sample sulit, abaikan yang sudah benar |
| **Class weight** (effective number) | Data imbalance | Neutral diberi bobot lebih besar secara proporsional |
| **KL regularizer** | Evidence sampah | Memaksa evidence kelas salah → 0 |
| **KL annealing** `λ_t` | Konvergensi prematur | Model punya waktu explore di awal training |
| **Auxiliary blending** `w_aux(t)` | Overfit, dead heads | Head belajar mandiri di awal, kolaborasi di akhir |

---

## 4. Training Strategy

### 4.1 Optimizer & Scheduler

```python
optimizer = AdamW([
    {'params': bert.parameters(),       'lr': 2e-5},
    {'params': resnet.parameters(),     'lr': 2e-5},
    {'params': coattn.parameters(),     'lr': 1e-3},
    {'params': edl_heads.parameters(),  'lr': 1e-3},
], weight_decay=0.01)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

### 4.2 Anti-Overfit Strategy

| Teknik | Implementasi | Kapan Aktif |
|--------|-------------|-------------|
| **Backbone freezing** | Freeze BERT & ResNet selama 3 epoch pertama | Epoch 0-2 |
| **Differential LR** | Backbone: 2e-5, Heads: 1e-3 (50× lebih besar) | Selalu |
| **Dropout** | 0.3 pada semua projection & MLP layers | Selalu |
| **Weight decay** | 0.01 via AdamW | Selalu |
| **KL annealing** | λ_t: 0→1 selama 10 epoch | Epoch 0-9 |
| **Aux weight decay** | w_aux: 1.0→0.1 seiring training | Seluruh training |
| **Focal loss** | γ=1.0, kurangi loss easy samples | Selalu |
| **Gradient clipping** | max_norm=1.0 | Selalu |
| **Early stopping** | Patience=7, monitor val Macro-F1 | Setelah epoch 10 |
| **Image augmentation** | RandomResizedCrop, HorizontalFlip, ColorJitter | Training only |

### 4.3 Training Loop

```python
best_f1 = 0
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    
    # Freeze backbone di awal
    if epoch < 3:
        for p in bert.parameters(): p.requires_grad = False
        for p in resnet.parameters(): p.requires_grad = False
    else:
        for p in bert.parameters(): p.requires_grad = True
        for p in resnet.parameters(): p.requires_grad = True

    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward
        H_v, H_t = extract_features(batch)
        H_v_prime, H_t_prime = co_attention(H_v, H_t)
        α_t, α_v, α_c = compute_heads(H_t, H_v, H_t_prime, H_v_prime)
        α_f = dempster_combine(α_t, α_v, α_c)
        
        # Loss
        y_onehot = F.one_hot(batch['label'], K).float()
        loss = total_loss(α_t, α_v, α_c, α_f, y_onehot,
                         class_weights, epoch, max_epochs, gamma=1.0)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    val_metrics = evaluate(model, val_loader)
    
    # Early stopping berdasarkan Macro-F1
    if val_metrics['macro_f1'] > best_f1:
        best_f1 = val_metrics['macro_f1']
        patience_counter = 0
        save_checkpoint(model, 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 7 and epoch > 10:
            print("Early stopping!")
            break
```

---

## 5. Hyperparameter

| Parameter | Value | Justifikasi |
|-----------|-------|-------------|
| Projection dim `d` | 256 | Balance capacity vs overfit untuk dataset kecil |
| Backbone LR | 2e-5 | Standard untuk fine-tuning pretrained models |
| Head LR | 1e-3 | Heads baru perlu belajar cepat |
| Weight decay | 0.01 | Standard AdamW regularization |
| Batch size | 32 | Disesuaikan GPU memory |
| Max epochs | 50 | Dengan early stopping |
| KL annealing | 10 epoch | Sesuai paper EDL original |
| Focal γ | 1.0 | Mild focal — cukup untuk dataset ini |
| Class weight β | 0.99 | Standard effective number |
| Dropout | 0.3 | Moderate — tidak terlalu agresif |
| Grad clip norm | 1.0 | Mencegah gradient explosion |
| Early stop patience | 7 | Setelah epoch 10 |
| Backbone freeze | 3 epoch | Warmup heads sebelum fine-tune backbone |

---

## 6. Evaluation Plan

### 6.1 Metrics

```
Primary:    Macro-F1 (memperlakukan semua kelas setara)
Secondary:  Weighted-F1, Per-class Precision/Recall/F1, Accuracy
```

### 6.2 Ablation Study (Wajib untuk Tesis)

| Eksperimen | Konfigurasi | Tujuan |
|------------|-------------|--------|
| Text-only | Hanya Head Text | Baseline unimodal teks |
| Image-only | Hanya Head Image | Baseline unimodal gambar |
| CoAttn-only | Hanya Head Co-Attention | Efek co-attention tanpa EDL fusion |
| Text + Image | DS_Combin(α_t, α_v) | Efek Dempster tanpa cross-modal head |
| Text + CoAttn | DS_Combin(α_t, α_c) | Apakah image head diperlukan? |
| Image + CoAttn | DS_Combin(α_v, α_c) | Apakah text head diperlukan? |
| **Full (3 Head)** | DS_Combin(α_t, α_v, α_c) | **Model lengkap** |
| Full tanpa focal | Tanpa focal weight | Apakah focal membantu? |
| Full tanpa class weight | Tanpa effective number | Apakah class weight membantu? |
| Full tanpa auxiliary | Hanya L_final | Apakah auxiliary loss membantu? |

### 6.3 Uncertainty Analysis

| Analisis | Metode | Tujuan |
|----------|--------|--------|
| Uncertainty per kelas | Box plot u_f per true label | Apakah neutral punya uncertainty tertinggi? |
| Uncertainty vs correctness | Scatter u_f vs correct/wrong | Apakah prediksi salah punya u tinggi? |
| Per-modality uncertainty | Distribusi u_t, u_v, u_c | Modalitas mana yang paling uncertain? |
| Accuracy vs uncertainty threshold | Plot acc saat filter u > τ | Apakah filtering uncertain samples meningkatkan acc? |
| Reliability diagram | Confidence vs accuracy | Apakah model well-calibrated? |

---

## 7. Kontribusi Tesis

| # | Kontribusi | Deskripsi |
|---|-----------|-----------|
| 1 | **3-Head EDL Architecture** | Tiga EDL head paralel (text, image, co-attention) yang menghasilkan evidence dan uncertainty dari perspektif berbeda |
| 2 | **Dempster's Fusion untuk Sentiment** | Penggunaan Dempster's combination rule untuk fusi evidence dari sumber unimodal dan cross-modal pada analisis sentimen |
| 3 | **Focal-EDL Loss** | Adaptasi focal loss ke dalam framework EDL untuk mengatasi class imbalance dan mencegah overfit pada easy samples |
| 4 | **Dynamic Auxiliary Blending** | Strategi auxiliary loss yang menurun seiring training untuk menyeimbangkan independent learning dan collaborative fine-tuning |
| 5 | **Uncertainty Analysis pada Sentiment** | Analisis mendalam tentang hubungan uncertainty dengan kelas neutral dan kualitas prediksi pada dataset multimodal |

---

## 8. Risiko & Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|--------|-------------|--------|----------|
| Dempster's conflict tinggi antar head | Medium | Prediksi tidak stabil | Monitor conflict coefficient C; investigasi jika > 0.5 |
| Focal γ terlalu tinggi → model abaikan terlalu banyak sample | Low | Underfitting | Mulai γ=1.0, naikkan hanya jika masih overfitting |
| BERT mendominasi → image head tidak belajar | High | u_v selalu tinggi | Auxiliary loss memaksa image head belajar; monitor u_v |
| Dataset MVSA terlalu kecil | Medium | Overfit persisten | Backbone freezing + dropout + early stopping |
