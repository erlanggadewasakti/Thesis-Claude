# ============================================================
# CELL 4: DATASET & DATALOADER
# ============================================================

# --- Hyperparameters ---
class HP:
    PROJ_DIM = 128           # Reduced: 256→128 to limit capacity on small dataset
    BACKBONE_LR = 5e-6       # Reduced: 2e-5→5e-6 to prevent backbone overfitting
    HEAD_LR = 5e-4           # Reduced: 1e-3→5e-4 for smoother convergence
    WEIGHT_DECAY = 0.02      # Increased: 0.01→0.02 for stronger L2 regularization
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    KL_ANNEALING_EPOCHS = 5  # Faster: 10→5 so KL regularizer kicks in earlier
    FOCAL_GAMMA = 2.0        # Increased: 1.0→2.0 to ignore easy samples harder
    CLASS_WEIGHT_BETA = 0.99
    DROPOUT = 0.5            # Increased: 0.3→0.5 for stronger regularization
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 10
    BACKBONE_FREEZE_EPOCHS = 8  # Extended: 3→8 to prevent early memorization
    MAX_TEXT_LEN = 128
    NUM_CLASSES = 3
    IMG_SIZE = 224

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

# --- Compute Effective Number Class Weights ---
class_counts = train_df["label"].value_counts().sort_index().values.astype(float)
beta = HP.CLASS_WEIGHT_BETA
effective_num = 1.0 - np.power(beta, class_counts)
weights = (1.0 - beta) / effective_num
class_weights = torch.tensor(
    weights / weights.sum() * HP.NUM_CLASSES, dtype=torch.float32
).to(device)
print(f"Class weights (effective number): {class_weights}")

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

train_loader = DataLoader(train_dataset, batch_size=HP.BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=HP.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=HP.BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f"\nDataLoaders ready. Train batches: {len(train_loader)}")
