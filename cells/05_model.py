# ============================================================
# CELL 5: MODEL ARCHITECTURE — UA-EDL-CoAttn v3
# ============================================================

# ----- Fix 3: Attention Pooling (replaces mean pool & CLS) -----
class AttentionPool(nn.Module):
    """
    Learned attention pooling over a sequence.
    Focuses on the most sentiment-relevant regions/tokens.
    Replaces mean pooling (which averages out sentiment signal).
    """
    def __init__(self, dim):
        super().__init__()
        self.attn_w = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, N, d)
        scores = self.attn_w(x)             # (B, N, 1)
        weights = F.softmax(scores, dim=1)  # (B, N, 1) — softmax over sequence
        return (weights * x).sum(dim=1)     # (B, d) — weighted sum


# ----- Stage 4: Dempster's Combination Rule -----
class DempsterCombination(nn.Module):
    """Combine multiple Dirichlet distributions via Dempster's rule."""
    def __init__(self, num_classes):
        super().__init__()
        self.K = num_classes

    def _combine_two(self, alpha1, alpha2):
        """Combine two Dirichlet parameter sets."""
        S1 = alpha1.sum(dim=1, keepdim=True)  # (B, 1)
        S2 = alpha2.sum(dim=1, keepdim=True)
        E1 = alpha1 - 1
        E2 = alpha2 - 1
        b1 = E1 / S1.expand_as(E1)
        b2 = E2 / S2.expand_as(E2)
        u1 = self.K / S1  # (B, 1)
        u2 = self.K / S2

        # Conflict coefficient
        bb = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))  # (B, K, K)
        bb_sum = bb.sum(dim=(1, 2))                         # (B,)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)  # (B,)
        C = (bb_sum - bb_diag).unsqueeze(1)                 # (B, 1)

        # Combined belief & uncertainty
        b_comb = (b1 * b2 + b1 * u2.expand_as(b1) + b2 * u1.expand_as(b2)) \
                 / (1 - C).expand_as(b1).clamp(min=1e-8)
        u_comb = (u1 * u2) / (1 - C).clamp(min=1e-8)       # (B, 1)

        # Back to Dirichlet params
        S_comb = self.K / u_comb.clamp(min=1e-8)
        e_comb = b_comb * S_comb.expand_as(b_comb)
        alpha_comb = e_comb + 1
        return alpha_comb

    def forward(self, alpha_list):
        """Combine a list of Dirichlet parameter tensors."""
        alpha = alpha_list[0]
        for i in range(1, len(alpha_list)):
            alpha = self._combine_two(alpha, alpha_list[i])
        return alpha


# ----- Stage 2: Co-Attention -----
class CoAttention(nn.Module):
    """Bidirectional co-attention between text and image features."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.W_a = nn.Linear(dim, dim, bias=False)  # text→image
        self.W_b = nn.Linear(dim, dim, bias=False)  # image→text
        self.scale = dim ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_t, H_v, text_mask=None):
        """
        H_t: (B, m, d) — text features
        H_v: (B, n, d) — image features (n=49 for ResNet)
        text_mask: (B, m) — 1 for real tokens, 0 for padding
        """
        # Text-guided visual attention: Q=text, K=V=image
        Q_t = self.W_a(H_t)                              # (B, m, d)
        scores_t2v = torch.bmm(Q_t, H_v.transpose(1, 2)) / self.scale  # (B, m, n)
        A_t2v = self.dropout(F.softmax(scores_t2v, dim=-1))
        H_v_prime = torch.bmm(A_t2v.transpose(1, 2), H_t)  # (B, n, d)

        # Image-guided textual attention: Q=image, K=V=text
        Q_v = self.W_b(H_v)                              # (B, n, d)
        scores_v2t = torch.bmm(Q_v, H_t.transpose(1, 2)) / self.scale  # (B, n, m)
        if text_mask is not None:
            mask = text_mask.unsqueeze(1).expand_as(scores_v2t)  # (B, n, m)
            scores_v2t = scores_v2t.masked_fill(mask == 0, -1e9)
        A_v2t = self.dropout(F.softmax(scores_v2t, dim=-1))
        H_t_prime = torch.bmm(A_v2t.transpose(1, 2), H_v)  # (B, m, d)

        return H_t_prime, H_v_prime


# ----- Stage 3: EDL Head -----
class EDLHead(nn.Module):
    """Evidential Deep Learning head: features → evidence → Dirichlet."""
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        evidence = F.relu(self.mlp(x))         # (B, K), non-negative
        alpha = evidence + 1                    # (B, K), >= 1
        S = alpha.sum(dim=1, keepdim=True)      # (B, 1)
        uncertainty = self.mlp[0].in_features    # just use K
        uncertainty = alpha.shape[1] / S        # u = K / S
        return alpha, uncertainty


# ----- Full Model -----
class UAEDLCoAttn(nn.Module):
    """
    Uncertainty-Aware EDL Co-Attention Model (v2).
    3 parallel EDL heads + Dempster's combination.
    """
    def __init__(self, num_classes=3, proj_dim=256, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        # --- Stage 1a: Image backbone (ResNet-50) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool+fc
        self.image_proj = nn.Sequential(
            nn.Linear(2048, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Stage 1b: Text backbone (BERT) ---
        self.text_backbone = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Stage 2: Co-Attention ---
        self.co_attention = CoAttention(proj_dim, dropout=dropout)

        # --- Stage 3a: Image EDL Head (from H_v, BEFORE co-attention) ---
        self.image_edl_head = EDLHead(proj_dim, proj_dim, num_classes, dropout)

        # --- Stage 3b: Co-Attention EDL Head (from H_v' + H_t', AFTER co-attention) ---
        self.coattn_edl_head = EDLHead(proj_dim * 2, proj_dim, num_classes, dropout)

        # --- Stage 3c: Text EDL Head (from H_t, BEFORE co-attention) ---
        self.text_edl_head = EDLHead(proj_dim, proj_dim, num_classes, dropout)

        # --- Stage 4: Dempster's Combination ---
        self.dempster = DempsterCombination(num_classes)

        # --- Fix 3: Attention Pooling (replaces mean pool / CLS) ---
        self.image_attn_pool = AttentionPool(proj_dim)
        self.text_attn_pool  = AttentionPool(proj_dim)

    def extract_image_features(self, images):
        """Stage 1a: Image → H_v (B, 49, d)"""
        feat = self.image_backbone(images)           # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 2048)
        H_v = self.image_proj(feat)                  # (B, 49, d)
        return H_v

    def extract_text_features(self, input_ids, attention_mask):
        """Stage 1b: Text → H_t (B, m, d)"""
        outputs = self.text_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        H_t_raw = outputs.last_hidden_state          # (B, m, 768)
        H_t = self.text_proj(H_t_raw)                # (B, m, d)
        return H_t

    def forward(self, input_ids, attention_mask, images):
        # === Stage 1: Feature Extraction ===
        H_v = self.extract_image_features(images)     # (B, 49, d)
        H_t = self.extract_text_features(input_ids, attention_mask)  # (B, m, d)

        # === Stage 2: Co-Attention ===
        H_t_prime, H_v_prime = self.co_attention(H_t, H_v, text_mask=attention_mask)

        # === Stage 3a: Image Head (BEFORE co-attention → unimodal) ===
        # Fix 3: Attention pool instead of mean pool
        h_v_pool = self.image_attn_pool(H_v)         # (B, d)
        alpha_v, u_v = self.image_edl_head(h_v_pool)

        # === Stage 3b: Co-Attention Head (AFTER co-attention → cross-modal) ===
        # Fix 3: Attention pool for both attended features
        h_v_att = self.image_attn_pool(H_v_prime)    # (B, d)
        h_t_att = self.text_attn_pool(H_t_prime)     # (B, d)
        h_cross = torch.cat([h_v_att, h_t_att], dim=-1)  # (B, 2d)
        alpha_c, u_c = self.coattn_edl_head(h_cross)

        # === Stage 3c: Text Head (BEFORE co-attention → unimodal) ===
        # Fix 3: Attention pool instead of CLS token
        h_t_cls = self.text_attn_pool(H_t)           # (B, d)
        alpha_t, u_t = self.text_edl_head(h_t_cls)

        # === Stage 4: Dempster's Combination ===
        # Unimodal first, then cross-modal
        alpha_f = self.dempster([alpha_t, alpha_v, alpha_c])

        return {
            "alpha_t": alpha_t, "u_t": u_t,
            "alpha_v": alpha_v, "u_v": u_v,
            "alpha_c": alpha_c, "u_c": u_c,
            "alpha_f": alpha_f,
        }

# --- Instantiate ---
model = UAEDLCoAttn(
    num_classes=HP.NUM_CLASSES,
    proj_dim=HP.PROJ_DIM,
    dropout=HP.DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
