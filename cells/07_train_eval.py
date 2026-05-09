# ============================================================
# CELL 7: TRAINING & EVALUATION FUNCTIONS
# ============================================================

def set_backbone_grad(model, requires_grad):
    """Toggle backbone gradient computation."""
    for p in model.image_backbone.parameters():
        p.requires_grad = requires_grad
    for p in model.text_backbone.parameters():
        p.requires_grad = requires_grad


def mixup_batch(images, y_onehot, alpha=0.4):
    """
    Mixup: interpolasi antar-sampel untuk augmentasi.
    Hanya untuk image — text tidak bisa di-mix (discrete tokens).
    """
    if alpha <= 0:
        return images, y_onehot

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Pastikan lam >= 0.5 (sample asli dominan)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]

    return mixed_images, mixed_y


def train_one_epoch(model, loader, optimizer, class_weights, epoch, max_epoch):
    model.train()

    # Backbone freezing for first N epochs
    if epoch < HP.BACKBONE_FREEZE_EPOCHS:
        set_backbone_grad(model, False)
    else:
        set_backbone_grad(model, True)

    use_rdrop = epoch >= HP.BACKBONE_FREEZE_EPOCHS  # R-Drop only after unfreeze

    total_loss_val = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        y_onehot = F.one_hot(labels, HP.NUM_CLASSES).float()

        # Fix 5: Mixup augmentation (50% chance per batch, only after backbone unfreeze)
        use_mixup = (epoch >= HP.BACKBONE_FREEZE_EPOCHS) and (random.random() < 0.5)
        if use_mixup:
            images, y_onehot = mixup_batch(images, y_onehot, alpha=HP.MIXUP_ALPHA)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, images)

        # R-Drop: second forward pass with different dropout masks
        outputs2 = None
        if use_rdrop:
            outputs2 = model(input_ids, attention_mask, images)

        loss = total_loss(
            outputs, y_onehot, class_weights,
            epoch, max_epoch,
            gamma=HP.FOCAL_GAMMA,
            kl_epochs=HP.KL_ANNEALING_EPOCHS,
            outputs2=outputs2
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), HP.GRAD_CLIP_NORM)
        optimizer.step()

        # Predictions from final alpha
        alpha_f = outputs["alpha_f"]
        S_f = alpha_f.sum(dim=1, keepdim=True)
        preds = (alpha_f / S_f).argmax(dim=1)

        total_loss_val += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss_val / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, acc


@torch.no_grad()
def evaluate(model, loader, class_weights, epoch, max_epoch, return_details=False):
    model.eval()

    total_loss_val = 0
    all_preds, all_labels = [], []
    all_uncertainties = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        y_onehot = F.one_hot(labels, HP.NUM_CLASSES).float()

        outputs = model(input_ids, attention_mask, images)

        loss = total_loss(
            outputs, y_onehot, class_weights,
            epoch, max_epoch,
            gamma=HP.FOCAL_GAMMA,
            kl_epochs=HP.KL_ANNEALING_EPOCHS
        )

        alpha_f = outputs["alpha_f"]
        S_f = alpha_f.sum(dim=1, keepdim=True)
        preds = (alpha_f / S_f).argmax(dim=1)
        u_f = HP.NUM_CLASSES / S_f.squeeze(1)

        total_loss_val += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_uncertainties.extend(u_f.cpu().numpy())

    avg_loss = total_loss_val / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)

    metrics = {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
    }

    if return_details:
        metrics["preds"] = np.array(all_preds)
        metrics["labels"] = np.array(all_labels)
        metrics["uncertainties"] = np.array(all_uncertainties)

    return metrics


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.should_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


print("Training & evaluation functions defined.")
