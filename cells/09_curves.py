# ============================================================
# CELL 9: TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Macro-F1
axes[1].plot(history["train_f1"], label="Train Macro-F1", linewidth=2)
axes[1].plot(history["val_f1"], label="Val Macro-F1", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Macro-F1")
axes[1].set_title("Macro-F1 Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Accuracy
axes[2].plot(history["train_acc"], label="Train Acc", linewidth=2)
axes[2].plot(history["val_acc"], label="Val Acc", linewidth=2)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Accuracy Curve")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
