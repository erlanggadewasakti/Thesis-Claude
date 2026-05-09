# ============================================================
# CELL 4: DATASET & DATALOADER
# ============================================================

# --- Hyperparameters ---
class HP:
    PROJ_DIM = 128           # Reduced: 256→128 to limit capacity on small dataset
    BACKBONE_LR = 5e-6       # Reduced: 2e-5→5e-6 to prevent backbone overfitting
    HEAD_LR = 5e-4           # Reduced: 1e-3→5e-4 for smoother convergence
    LR_DECAY = 0.85          # Layerwise LR decay factor for BERT
    WEIGHT_DECAY = 0.02      # Increased: 0.01→0.02 for stronger L2 regularization
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    KL_ANNEALING_EPOCHS = 5  # KL regularizer kicks in earlier
    FOCAL_GAMMA = 2.0        # Ignore easy samples harder
    DROPOUT = 0.5            # Stronger regularization
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 10
    BACKBONE_FREEZE_EPOCHS = 8  # Extended: 3→8 to prevent early memorization
    MAX_TEXT_LEN = 128
    NUM_CLASSES = 3
    IMG_SIZE = 224
    MIXUP_ALPHA = 0.4        # Mixup interpolation strength
    NEUTRAL_BOOST = 2.0      # Extra loss multiplier for neutral samples

# --- Train/Val/Test Split ---
train_df, test_df = train_test_split(
    df, test_size=CFG.TEST_SIZE, random_state=42, stratify=df["label"]
)
train_df, val_df = train_test_split(
    train_df, test_size=CFG.VAL_SIZE / (1 - CFG.TEST_SIZE),
    random_state=42, stratify=train_df["label"]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train label distribution:\n{train_df['label'].value_counts().sort_index()}")

# --- Fix 1: Inverse Sqrt Class Weights ---
# Effective number with beta=0.99 produced near-flat weights [0.99, 1.03, 0.99]
# Inverse sqrt gives more discriminative weights: neg≈0.88, neu≈1.50, pos≈0.63
class_counts = train_df["label"].value_counts().sort_index().values.astype(float)
weights = 1.0 / np.sqrt(class_counts)
class_weights = torch.tensor(
    weights / weights.sum() * HP.NUM_CLASSES, dtype=torch.float32
).to(device)
print(f"Class weights (inverse sqrt): {class_weights}")
print(f"  neg={class_weights[0]:.4f}, neu={class_weights[1]:.4f}, pos={class_weights[2]:.4f}")

# --- Image Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((HP.IMG_SIZE + 32, HP.IMG_SIZE + 32)),
    transforms.RandomResizedCrop(HP.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((HP.IMG_SIZE, HP.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Tokenizer ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- Dataset Class ---
class MVSADataset(Dataset):
    def __init__(self, dataframe, transform, tokenizer, max_len=HP.MAX_TEXT_LEN):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Text ---
        text = str(row["text"]) if row["text"] else ""
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # --- Image ---
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, HP.IMG_SIZE, HP.IMG_SIZE)

        # --- Label ---
        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

# --- DataLoaders ---
train_dataset = MVSADataset(train_df, train_transform, tokenizer)
val_dataset = MVSADataset(val_df, val_transform, tokenizer)
test_dataset = MVSADataset(test_df, val_transform, tokenizer)

# --- Fix 2: WeightedRandomSampler to oversample minority classes ---
# Neutral (10%) gets 5.7× more samples, negative (30%) gets 2.0× more
from torch.utils.data import WeightedRandomSampler
_max_count = float(class_counts.max())  # = 1878 (positive)
label_to_sample_w = {
    0: _max_count / class_counts[0],  # neg: 1878/950 ≈ 1.98×
    1: _max_count / class_counts[1],  # neu: 1878/329 ≈ 5.71× oversample!
    2: 1.0,                           # pos: baseline
}
sample_weights = [label_to_sample_w[l] for l in train_df["label"].values]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=HP.BATCH_SIZE,
    sampler=sampler,  # replaces shuffle=True
    num_workers=0, pin_memory=True, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=HP.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=HP.BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f"\nDataLoaders ready. Train batches: {len(train_loader)}")
print(f"WeightedRandomSampler: neg×{label_to_sample_w[0]:.1f}, neu×{label_to_sample_w[1]:.1f}, pos×{label_to_sample_w[2]:.1f}")
