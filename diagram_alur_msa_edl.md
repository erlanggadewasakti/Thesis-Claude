# Diagram Alur Implementasi MSA EDL

Dokumen ini menjelaskan alur implementasi notebook `claude_msa_edl.ipynb` untuk analisis sentimen multimodal MVSA-Single menggunakan RoBERTa, ResNet50, feature decoupling, bi-directional cross-attention, Evidential Deep Learning, dan Dempster-Shafer fusion.

## Diagram Mermaid

```mermaid
flowchart TD
    A["MVSA-Single Dataset<br/>labelResultAllFinal.txt<br/>data/*.txt + data/*.jpg"] --> B["Template Loader Dataset<br/>filter label konflik<br/>map label: negative=0, neutral=1, positive=2<br/>load text multi-encoding<br/>buat image_path"]

    B --> C["Stratified Split<br/>Train 80%<br/>Validation 10%<br/>Test 10%"]
    C --> D["MVSADataset + DataLoader"]

    D --> E1["Text Pipeline<br/>AutoTokenizer roberta-base<br/>input_ids + attention_mask"]
    D --> E2["Image Pipeline<br/>Resize/Crop 224x224<br/>ImageNet Normalize<br/>zero tensor fallback"]

    E1 --> F1["Text Encoder<br/>RoBERTa last_hidden_state"]
    E2 --> F2["Image Encoder<br/>ResNet50 conv feature map<br/>spatial image tokens"]

    F1 --> G1["Text Projection<br/>shared dimension d"]
    F2 --> G2["Image Projection<br/>shared dimension d"]

    G1 --> H1["Text Feature Decoupling<br/>sentiment feature<br/>background feature"]
    G2 --> H2["Image Feature Decoupling<br/>sentiment feature<br/>background feature"]

    H1 --> O["Orthogonal Loss<br/>sentiment and background<br/>dibuat tidak saling tumpang tindih"]
    H2 --> O

    H1 --> I1["Text-to-Image BCA<br/>Q = text sentiment<br/>K,V = image sentiment"]
    H2 --> I1

    H2 --> I2["Image-to-Text BCA<br/>Q = image sentiment<br/>K,V = text sentiment"]
    H1 --> I2

    I1 --> J1["EDL Head T2V<br/>evidence = Softplus Linear<br/>alpha = evidence + 1<br/>belief + uncertainty"]
    I2 --> J2["EDL Head V2T<br/>evidence = Softplus Linear<br/>alpha = evidence + 1<br/>belief + uncertainty"]

    J1 --> L1["EDL Opinion T2V<br/>alpha_t2v, belief_t2v, u_t2v"]
    J2 --> L2["EDL Opinion V2T<br/>alpha_v2t, belief_v2t, u_v2t"]

    L1 --> M["Dempster-Shafer Fusion<br/>combine belief masses<br/>estimate conflict<br/>produce alpha_fused"]
    L2 --> M

    M --> N["Final Prediction<br/>preds = argmax belief_fused<br/>u_fused for uncertainty analysis"]

    N --> P["Training Objective<br/>EDL Expected MSE<br/>KL annealing<br/>auxiliary directional loss<br/>orthogonal regularization"]
    O --> P

    P --> Q["Optimizer + Scheduler<br/>AdamW<br/>CosineAnnealingLR<br/>gradient clipping"]

    Q --> R["Validation<br/>Accuracy<br/>Macro-F1<br/>Weighted-F1<br/>mean conflict<br/>uncertainty correct vs wrong"]

    R --> S["Best Model State<br/>selected by validation Macro-F1<br/>EarlyStopping"]
    S --> T["Test Evaluation<br/>classification report<br/>confusion matrix<br/>uncertainty distribution"]
```

## Penjelasan Alur

1. **Dataset dan preprocessing awal**
   Dataset MVSA-Single dimuat dari `labelResultAllFinal.txt` dan folder `data`. Template loader yang sudah ada membersihkan pasangan label teks-gambar yang saling bertentangan, memetakan label ke tiga kelas, membaca file teks dengan beberapa opsi encoding, dan membuat `image_path` untuk setiap sampel.

2. **Split dan DataLoader**
   Data dibagi secara stratified menjadi train, validation, dan test dengan rasio 80:10:10. `MVSADataset` menghasilkan `input_ids`, `attention_mask`, tensor gambar, label, dan `sample_id`. Pipeline gambar memakai transform train/eval terpisah, termasuk normalisasi ImageNet dan fallback zero tensor bila gambar gagal dibaca.

3. **Ekstraksi fitur unimodal**
   Modalitas teks diproses oleh `roberta-base` dan memakai `last_hidden_state` sebagai token teks. Modalitas gambar diproses oleh ResNet50 hingga feature map konvolusional, lalu feature map diubah menjadi token spasial gambar. Kedua modalitas diproyeksikan ke dimensi bersama agar dapat diproses oleh attention layer yang sama.

4. **Feature decoupling**
   Setiap modalitas dipisahkan menjadi fitur sentimen dan fitur background menggunakan MLP terpisah. Fitur sentimen diteruskan ke cross-attention, sedangkan fitur background digunakan untuk regularisasi orthogonal. Tujuannya adalah mengurangi noise dan mendorong model menangkap sinyal sentimen yang lebih bersih.

5. **Bi-directional Cross-Attention**
   Model memakai dua arah attention. Pada alur Text-to-Image, token sentimen teks menjadi query untuk menyoroti token sentimen gambar. Pada alur Image-to-Text, token sentimen gambar menjadi query untuk menyoroti token sentimen teks. Dua arah ini membantu model membaca hubungan multimodal tanpa memaksa salah satu modalitas selalu dominan.

6. **Evidential Deep Learning**
   Setiap alur attention masuk ke EDL head satu kali. EDL head menghasilkan evidence non-negatif dengan `Softplus`, lalu membentuk parameter Dirichlet `alpha = evidence + 1`. Dari `alpha`, model menghitung belief dan uncertainty untuk masing-masing alur tanpa melakukan manual feature gating ulang.

7. **Dempster-Shafer fusion**
   Dua opini evidential dari alur Text-to-Image dan Image-to-Text digabung menggunakan Dempster-Shafer fusion. Fusion ini menghitung massa belief gabungan, uncertainty gabungan, dan konflik antar alur, sehingga pembobotan berbasis ketidakpastian terjadi pada tahap fusion. Prediksi akhir diambil dari kelas dengan `belief_fused` tertinggi.

8. **Loss dan optimisasi**
   Training memakai EDL Expected MSE sebagai loss klasifikasi utama, KL divergence dengan annealing untuk mengontrol evidence berlebihan, auxiliary loss untuk dua directional head, serta orthogonal loss kecil untuk menjaga decoupling. Optimizer menggunakan AdamW, scheduler menggunakan CosineAnnealingLR, dan training loop memakai gradient clipping.

9. **Evaluasi**
   Evaluasi mencatat accuracy, Macro-F1, Weighted-F1, mean conflict, serta rata-rata uncertainty pada prediksi benar dan salah. Test evaluation menampilkan classification report, confusion matrix, dan distribusi uncertainty agar performa model dapat dianalisis dari sisi akurasi dan kalibrasi ketidakpastian.

## Ringkasan Output Model

`DecoupledBCAEDL.forward(...)` mengembalikan output utama berikut:

- `alpha_t2v`, `belief_t2v`, `u_t2v`: opini evidential dari alur Text-to-Image.
- `alpha_v2t`, `belief_v2t`, `u_v2t`: opini evidential dari alur Image-to-Text.
- `alpha_fused`, `belief_fused`, `u_fused`: opini final setelah Dempster-Shafer fusion.
- `preds`: prediksi kelas akhir berdasarkan `belief_fused`.
- `orthogonal_loss`: regularisasi feature decoupling.
- `conflict`: estimasi konflik antara dua alur evidential.
