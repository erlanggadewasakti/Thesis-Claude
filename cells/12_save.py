# ============================================================
# CELL 12: SAVE MODEL
# ============================================================

save_path = os.path.join(os.path.dirname(CFG.ROOT_DIR), "ua_edl_coattn_best.pt")

torch.save({
    "model_state_dict": model.state_dict(),
    "hyperparameters": {
        "proj_dim": HP.PROJ_DIM,
        "num_classes": HP.NUM_CLASSES,
        "dropout": HP.DROPOUT,
    },
    "class_weights": class_weights.cpu(),
    "history": history,
    "best_val_macro_f1": early_stopping.best_score,
    "test_metrics": {
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
    },
}, save_path)

print(f"✅ Model saved to: {save_path}")
print(f"   Best Val Macro-F1: {early_stopping.best_score:.4f}")
print(f"   Test Macro-F1:     {test_metrics['macro_f1']:.4f}")
