# ============================================================
# CELL 10: TEST SET EVALUATION
# ============================================================

test_metrics = evaluate(
    model, test_loader, class_weights,
    epoch=HP.MAX_EPOCHS, max_epoch=HP.MAX_EPOCHS,
    return_details=True
)

print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"Macro-F1:    {test_metrics['macro_f1']:.4f}")
print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")
print()

# Classification Report
print(classification_report(
    test_metrics["labels"], test_metrics["preds"],
    target_names=["negative", "neutral", "positive"],
    digits=4
))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"],
            ax=ax, annot_kws={"size": 14})
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title("Confusion Matrix — UA-EDL-CoAttn", fontsize=14)
plt.tight_layout()
plt.show()
