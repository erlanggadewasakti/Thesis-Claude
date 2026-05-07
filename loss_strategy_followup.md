# Jawaban: Gradient Flow & Class Weight Strategy

---

## Pertanyaan 1: Mengapa 4 loss hanya update layer masing-masing?

### Klarifikasi Penting: Ini Bukan by Design, tapi by Computational Graph

Diagram sebelumnya bisa menyesatkan. Yang terjadi sebenarnya:

```
L_text  → backprop → text EDL → text projection → BERT
                     ↑
                     HANYA layer ini yang ada di computational path α_t
                     ResNet TIDAK terlibat dalam menghitung α_t
                     → maka gradient L_text TIDAK BISA sampai ke ResNet
```

Ini bukan pilihan desain — ini **konsekuensi matematis** dari computational graph. Gradient hanya bisa flow melalui operasi yang terlibat dalam menghitung output.

**TAPI**: `L_final` MEMANG sudah update SEMUA, karena:

```
α_final = DS_Combin(α_t, α_v, α_c)
                     ↑     ↑     ↑
                     │     │     └── terhubung ke co-attention → BERT + ResNet
                     │     └── terhubung ke ResNet
                     └── terhubung ke BERT

L_final → backprop → α_final → Dempster → α_t → BERT
                                         → α_v → ResNet  
                                         → α_c → Co-Attention → BERT + ResNet
```

Jadi **L_final sudah melakukan apa yang Anda minta** — update seluruh network sekaligus.

### Lalu Mengapa Masih Perlu L_text, L_image, L_coattn?

**Tanpa auxiliary loss (hanya L_final):**

```
Gradient: L_final → Dempster → 3 head → semua layer
```

Masalahnya:
1. Gradient harus melewati **Dempster's combination** — operasi non-linear yang bisa menyebabkan **vanishing gradient** ke head yang uncertain
2. Jika satu head mendominasi (misal text selalu paling certain), Dempster akan "mempercayai" text → gradient ke image dan co-attention **sangat kecil** → image head tidak belajar
3. Tidak ada jaminan setiap head bisa berdiri sendiri → model jadi fragile

**Dengan auxiliary loss:**

```
Gradient ke BERT: dari L_text (langsung) + dari L_final (via Dempster) + dari L_coattn (via co-attention)
                  = gradient signal KUAT dan STABIL
```

Auxiliary loss memastikan:
- Setiap head **dipaksa** belajar, bahkan jika Dempster memberi gradient kecil
- Setiap head bisa berdiri sendiri → **regularisasi** (model tidak bisa cheat)
- Gradient ke backbone lebih stabil → **training lebih smooth, kurang overfit**

### Perbandingan 3 Pendekatan

| Pendekatan | Deskripsi | Anti-Overfit | Masalah |
|------------|-----------|-------------|---------|
| **A: Concat semua** | `h = concat(h_t, h_v, h_c)` → 1 EDL head → 1 loss | ⭐⭐ Rendah | Satu loss untuk semua → model bisa overfit ke shortcut apapun; kehilangan Dempster's rule |
| **B: 4 loss isolated** | Setiap head punya loss sendiri, tidak ada sharing | ⭐⭐⭐⭐ Tinggi | Head tidak belajar berkolaborasi; over-regularize → underfitting |
| **C: Hybrid (saat ini)** | 3 auxiliary + 1 final (L_final update semua) | ⭐⭐⭐⭐⭐ Optimal | — |

### Rekomendasi: Tetap Hybrid, tapi dengan Gradient Blending

Jika Anda khawatir auxiliary loss masih menyebabkan overfit, gunakan **dynamic weight** yang menurun seiring training:

```python
def compute_loss(alpha_t, alpha_v, alpha_c, alpha_final, y, cw, epoch, max_epoch):
    L_text   = head_loss(alpha_t, y, cw, epoch)
    L_image  = head_loss(alpha_v, y, cw, epoch)
    L_coattn = head_loss(alpha_c, y, cw, epoch)
    L_final  = head_loss(alpha_final, y, cw, epoch)
    
    # Auxiliary weight menurun seiring training
    # Awal: auxiliary kuat (regularisasi, bantu setiap head belajar)
    # Akhir: auxiliary rendah (biarkan L_final dominasi fine-tuning)
    aux_w = max(0.1, 1.0 - epoch / max_epoch)  # 1.0 → 0.1
    
    return aux_w * (L_text + L_image + L_coattn) + 1.0 * L_final
```

**Intuisi**: Di awal training, auxiliary loss membantu setiap head belajar representasi yang baik. Di akhir training, L_final mendominasi untuk fine-tune kolaborasi antar head via Dempster.

---

## Pertanyaan 2: Apakah class weight inverse frequency optimal?

### Jawaban Singkat: Tidak selalu.

Inverse frequency adalah **starting point yang reasonable**, tapi punya kelemahan:

### Kelemahan Inverse Frequency

```
w_neg = 1.106, w_neu = 3.224, w_pos = 0.560
```

**Masalah 1**: Linear scaling mungkin **tidak cukup agresif** untuk extreme imbalance.
- Neutral hanya 70 vs 403 positive (rasio 1:5.8)
- Weight 3.224 vs 0.560 (rasio 5.8:1) — ini **linear compensation**
- Tapi learning dynamics tidak linear! Model yang sudah bias ke positive membutuhkan koreksi **lebih dari linear** untuk belajar neutral

**Masalah 2**: Semua sample neutral dianggap sama pentingnya.
- Ada sample neutral yang "mudah" (kedua modalitas jelas neutral)
- Ada sample neutral yang "sulit" (ambigu, dekat boundary)
- Inverse frequency memberi bobot sama ke semua → model bisa overfit ke yang mudah, masih gagal di yang sulit

### 3 Alternatif yang Lebih Robust

#### Alternatif 1: Effective Number of Samples (Cui et al., CVPR 2019)

```python
# β biasanya 0.9, 0.99, atau 0.999
beta = 0.99
effective_num = 1.0 - beta ** class_counts          # [1-0.99^204, 1-0.99^70, 1-0.99^403]
weights = (1.0 - beta) / effective_num
weights = weights / weights.sum() * K                # normalize

# Contoh dengan β=0.99:
# effective_num = [0.8697, 0.5043, 0.9828]
# weights_raw = [0.0115, 0.0198, 0.0102]
# weights_norm = [0.831, 1.430, 0.736]
```

**Kelebihan**: Tidak over-compensate kelas minoritas seperti inverse frequency. Lebih stabil.

#### Alternatif 2: Focal-Style Weighting dalam EDL

Alih-alih weight statis per kelas, gunakan **weight dinamis per sample** berdasarkan seberapa sulit sample tersebut:

```python
def focal_edl_loss(alpha, y_onehot, gamma=2.0):
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    
    # Probabilitas kelas benar
    p_correct = (p_hat * y_onehot).sum(dim=1, keepdim=True)  # (batch, 1)
    
    # Focal weight: sample yang sudah benar → weight kecil
    #               sample yang masih salah → weight besar
    focal_weight = (1 - p_correct) ** gamma    # (batch, 1)
    
    # Standard EDL loss
    err = (y_onehot - p_hat) ** 2
    var = p_hat * (1 - p_hat) / (S + 1)
    
    # Apply focal weight per sample (bukan per kelas)
    loss = focal_weight * (err + var)
    
    return loss.sum(dim=1).mean()
```

**Intuisi**:
- Sample positive yang mudah (p_correct=0.9): focal_weight = 0.01 → hampir diabaikan
- Sample neutral yang sulit (p_correct=0.2): focal_weight = 0.64 → gradient besar
- Model **otomatis fokus pada sample yang sulit** tanpa perlu class weight statis

**Mengapa ini membantu overfit?**: Model berhenti "mengulang" sample yang sudah benar (easy positives), dan mengalokasikan kapasitasnya untuk sample yang masih salah (hard neutrals).

#### Alternatif 3: Adaptive Class Weight (Learned)

```python
# Class weight sebagai learnable parameter
log_class_weights = nn.Parameter(torch.zeros(K))  # (K,)

def adaptive_weighted_loss(alpha, y_onehot, log_class_weights):
    # Softmax agar weights tetap positif dan normalized
    class_weights = F.softmax(log_class_weights, dim=0) * K
    
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    err = (y_onehot - p_hat) ** 2
    var = p_hat * (1 - p_hat) / (S + 1)
    
    loss = class_weights * (err + var)
    return loss.sum(dim=1).mean()
```

**Kelebihan**: Model sendiri yang menentukan bobot optimal per kelas.
**Risiko**: Bisa collapse (semua weight sama). Perlu constraint atau regularisasi.

### Rekomendasi: Kombinasi Effective Number + Focal

```python
def combined_edl_loss(alpha, y_onehot, class_weights_eff, gamma=1.0):
    """
    class_weights_eff: dari effective number of samples
    gamma: focal exponent (1.0 = mild, 2.0 = aggressive)
    """
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    
    # Focal weight (per sample)
    p_correct = (p_hat * y_onehot).sum(dim=1, keepdim=True)
    focal_weight = (1 - p_correct) ** gamma
    
    # EDL components
    err = (y_onehot - p_hat) ** 2
    var = p_hat * (1 - p_hat) / (S + 1)
    
    # Double weighting: class weight (statis) × focal weight (dinamis)
    loss = focal_weight * class_weights_eff * (err + var)
    
    return loss.sum(dim=1).mean()
```

**Mengapa kombinasi?**
- **Class weight** (statis): Memastikan neutral tidak diabaikan secara struktural
- **Focal weight** (dinamis): Memastikan sample yang sudah benar tidak di-overfit

| Masalah | Class Weight Saja | Focal Saja | Kombinasi |
|---------|-------------------|------------|-----------|
| Imbalance | ✅ Teratasi | ❌ Tidak langsung | ✅ Teratasi |
| Overfit ke easy samples | ❌ Masih bisa | ✅ Teratasi | ✅ Teratasi |
| Hard neutral samples | ⚠️ Partial | ✅ Teratasi | ✅ Teratasi |
