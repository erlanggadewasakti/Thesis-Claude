# ============================================================
# CELL 8: TRAINING LOOP
# ============================================================

# --- Fix 6: Layerwise LR decay for BERT ---
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
    get_layerwise_lr_params(model, bert_base_lr=HP.BACKBONE_LR, lr_decay=HP.LR_DECAY, head_lr=HP.HEAD_LR),
    weight_decay=HP.WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

early_stopping = EarlyStopping(patience=HP.EARLY_STOP_PATIENCE, min_delta=0.001)

# --- History ---
history = {
    "train_loss": [], "train_f1": [], "train_acc": [],
    "val_loss": [], "val_f1": [], "val_acc": [], "val_wf1": [],
}

# --- Training ---
print("=" * 60)
print("TRAINING START")
print("=" * 60)

for epoch in range(HP.MAX_EPOCHS):
    # Train
    train_loss, train_f1, train_acc = train_one_epoch(
        model, train_loader, optimizer, class_weights, epoch, HP.MAX_EPOCHS
    )

    # Validate
    val_metrics = evaluate(
        model, val_loader, class_weights, epoch, HP.MAX_EPOCHS
    )

    scheduler.step()

    # Log
    history["train_loss"].append(train_loss)
    history["train_f1"].append(train_f1)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_metrics["loss"])
    history["val_f1"].append(val_metrics["macro_f1"])
    history["val_acc"].append(val_metrics["accuracy"])
    history["val_wf1"].append(val_metrics["weighted_f1"])

    freeze_str = "❄️ FROZEN" if epoch < HP.BACKBONE_FREEZE_EPOCHS else "🔥 FINE-TUNE"
    w_aux = max(0.1, 1.0 - epoch / HP.MAX_EPOCHS)

    print(f"Epoch {epoch+1:>3}/{HP.MAX_EPOCHS} | {freeze_str} | w_aux={w_aux:.2f} | "
          f"Train L={train_loss:.4f} F1={train_f1:.4f} | "
          f"Val L={val_metrics['loss']:.4f} F1={val_metrics['macro_f1']:.4f} "
          f"wF1={val_metrics['weighted_f1']:.4f} Acc={val_metrics['accuracy']:.4f}")

    # Early stopping (only after KL annealing is done)
    if epoch >= HP.KL_ANNEALING_EPOCHS:
        early_stopping(val_metrics["macro_f1"], model)
        if early_stopping.should_stop:
            print(f"\n⏹ Early stopping at epoch {epoch+1}. "
                  f"Best val Macro-F1: {early_stopping.best_score:.4f}")
            break

# Load best model
if early_stopping.best_model is not None:
    model.load_state_dict(early_stopping.best_model)
    print("✅ Loaded best model weights.")
else:
    print("⚠️ No early stopping triggered, using last epoch weights.")

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
