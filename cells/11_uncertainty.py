# ============================================================
# CELL 11: UNCERTAINTY ANALYSIS
# ============================================================

@torch.no_grad()
def collect_uncertainty_details(model, loader):
    """Collect per-head uncertainties for analysis."""
    model.eval()
    results = {"u_t": [], "u_v": [], "u_c": [], "u_f": [],
               "preds": [], "labels": [], "correct": []}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, images)

        alpha_f = outputs["alpha_f"]
        S_f = alpha_f.sum(dim=1, keepdim=True)
        preds = (alpha_f / S_f).argmax(dim=1)
        u_f = HP.NUM_CLASSES / S_f.squeeze(1)

        results["u_t"].extend(outputs["u_t"].squeeze(1).cpu().numpy())
        results["u_v"].extend(outputs["u_v"].squeeze(1).cpu().numpy())
        results["u_c"].extend(outputs["u_c"].squeeze(1).cpu().numpy())
        results["u_f"].extend(u_f.cpu().numpy())
        results["preds"].extend(preds.cpu().numpy())
        results["labels"].extend(labels.cpu().numpy())
        results["correct"].extend((preds == labels).cpu().numpy())

    return {k: np.array(v) for k, v in results.items()}

unc_data = collect_uncertainty_details(model, test_loader)

# --- Plot 1: Uncertainty Distribution per True Class ---
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
head_names = ["Text (u_t)", "Image (u_v)", "Co-Attn (u_c)", "Final (u_f)"]
head_keys = ["u_t", "u_v", "u_c", "u_f"]
class_names = ["negative", "neutral", "positive"]

for ax, key, name in zip(axes, head_keys, head_names):
    data_by_class = [unc_data[key][unc_data["labels"] == c] for c in range(3)]
    bp = ax.boxplot(data_by_class, labels=class_names, patch_artist=True)
    colors = ["#FF6B6B", "#FFA726", "#66BB6A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(name, fontsize=12)
    ax.set_ylabel("Uncertainty")
    ax.grid(True, alpha=0.3)

plt.suptitle("Uncertainty Distribution per True Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# --- Plot 2: Uncertainty vs Correctness ---
fig, ax = plt.subplots(figsize=(8, 5))
correct_u = unc_data["u_f"][unc_data["correct"] == 1]
wrong_u = unc_data["u_f"][unc_data["correct"] == 0]

ax.hist(correct_u, bins=30, alpha=0.6, label=f"Correct (n={len(correct_u)})",
        color="#66BB6A", edgecolor="black")
ax.hist(wrong_u, bins=30, alpha=0.6, label=f"Wrong (n={len(wrong_u)})",
        color="#FF6B6B", edgecolor="black")
ax.set_xlabel("Final Uncertainty (u_f)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Uncertainty: Correct vs Wrong Predictions", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot 3: Accuracy vs Uncertainty Threshold ---
thresholds = np.linspace(0.01, 1.0, 50)
accs_at_threshold = []
coverage = []
for t in thresholds:
    mask = unc_data["u_f"] <= t
    if mask.sum() > 0:
        accs_at_threshold.append(accuracy_score(
            unc_data["labels"][mask], unc_data["preds"][mask]
        ))
        coverage.append(mask.mean())
    else:
        accs_at_threshold.append(0)
        coverage.append(0)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(thresholds, accs_at_threshold, color="#1976D2", linewidth=2, label="Accuracy")
ax2.plot(thresholds, coverage, color="#FF9800", linewidth=2, linestyle="--", label="Coverage")
ax1.set_xlabel("Uncertainty Threshold", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12, color="#1976D2")
ax2.set_ylabel("Coverage (fraction of data)", fontsize=12, color="#FF9800")
ax1.set_title("Accuracy vs Uncertainty Threshold", fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Stats ---
print("=" * 60)
print("UNCERTAINTY STATISTICS")
print("=" * 60)
for key, name in zip(head_keys, head_names):
    print(f"\n{name}:")
    for c, cn in enumerate(class_names):
        vals = unc_data[key][unc_data["labels"] == c]
        print(f"  {cn:>10}: mean={vals.mean():.4f}, median={np.median(vals):.4f}, std={vals.std():.4f}")
print(f"\nCorrect predictions — mean u_f: {correct_u.mean():.4f}")
print(f"Wrong predictions   — mean u_f: {wrong_u.mean():.4f}")
