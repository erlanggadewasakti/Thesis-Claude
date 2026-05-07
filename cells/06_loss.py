# ============================================================
# CELL 6: LOSS FUNCTIONS — Focal-EDL + KL Regularizer
# ============================================================

def focal_edl_loss(alpha, y_onehot, class_weights, gamma=1.0):
    """
    Focal-weighted EDL loss (SSE Bayes Risk).
    - class_weights: static per-class weight (effective number)
    - focal: dynamic per-sample weight based on prediction confidence
    """
    S = alpha.sum(dim=1, keepdim=True)             # (B, 1)
    p_hat = alpha / S                               # (B, K)

    # Focal weight: (1 - p_correct)^gamma
    p_correct = (p_hat * y_onehot).sum(dim=1, keepdim=True)  # (B, 1)
    focal_w = (1 - p_correct.detach()) ** gamma     # (B, 1) — detach to avoid instability

    # EDL components
    err = (y_onehot - p_hat) ** 2                   # (B, K)
    var = p_hat * (1 - p_hat) / (S + 1)             # (B, K)

    # Combined weighting: focal (per-sample) × class_weight (per-class)
    loss = focal_w * class_weights.unsqueeze(0) * (err + var)  # (B, K)

    return loss.sum(dim=1).mean()


def kl_divergence_reg(alpha, y_onehot, epoch, annealing_epochs=10):
    """
    KL divergence regularizer: penalizes misleading evidence.
    Pushes evidence for WRONG classes toward zero.
    """
    K = alpha.shape[1]
    lambda_t = min(1.0, epoch / annealing_epochs)

    # Remove evidence from correct class (don't penalize it)
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha  # correct class → 1

    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)    # (B, 1)

    # KL[Dir(alpha_tilde) || Dir(1, 1, ..., 1)]
    ln_B = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
    ln_B_uni = torch.lgamma(torch.ones(1, K, device=alpha.device)).sum() \
             - torch.lgamma(torch.tensor(float(K), device=alpha.device))

    dg0 = torch.digamma(S_tilde)
    dg1 = torch.digamma(alpha_tilde)

    kl = ((alpha_tilde - 1) * (dg1 - dg0)).sum(dim=1, keepdim=True) + ln_B + ln_B_uni

    return lambda_t * kl.mean()


def head_loss(alpha, y_onehot, class_weights, epoch, gamma=1.0, kl_epochs=10):
    """Combined loss for a single EDL head."""
    return focal_edl_loss(alpha, y_onehot, class_weights, gamma) \
         + kl_divergence_reg(alpha, y_onehot, epoch, kl_epochs)


def total_loss(outputs, y_onehot, class_weights, epoch, max_epoch,
               gamma=1.0, kl_epochs=10, outputs2=None):
    """
    Total multi-task loss with dynamic auxiliary blending.
    L = w_aux * (L_text + L_image + L_coattn) + 1.0 * L_final + rdrop
    If outputs2 is provided, adds R-Drop consistency regularization.
    """
    L_t = head_loss(outputs["alpha_t"], y_onehot, class_weights, epoch, gamma, kl_epochs)
    L_v = head_loss(outputs["alpha_v"], y_onehot, class_weights, epoch, gamma, kl_epochs)
    L_c = head_loss(outputs["alpha_c"], y_onehot, class_weights, epoch, gamma, kl_epochs)
    L_f = head_loss(outputs["alpha_f"], y_onehot, class_weights, epoch, gamma, kl_epochs)

    # Dynamic auxiliary weight: strong early (regularize), weak late (fine-tune)
    w_aux = max(0.1, 1.0 - epoch / max_epoch)

    loss = w_aux * (L_t + L_v + L_c) + 1.0 * L_f

    # R-Drop consistency regularization
    if outputs2 is not None:
        def rdrop_kl(a1, a2):
            s1, s2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
            p1, p2 = a1 / s1, a2 / s2
            kl_12 = (p1 * (p1.log() - p2.log())).sum(1).mean()
            kl_21 = (p2 * (p2.log() - p1.log())).sum(1).mean()
            return (kl_12 + kl_21) / 2

        rdrop = rdrop_kl(outputs["alpha_f"], outputs2["alpha_f"])
        loss = loss + 0.5 * rdrop

    return loss
