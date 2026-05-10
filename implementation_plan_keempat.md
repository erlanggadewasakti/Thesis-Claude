# Overfitting Mitigation for Multimodal Sentiment Analysis Model

The model is currently suffering from severe overfitting. Training Macro-F1 reaches ~0.97 while validation Macro-F1 plateaus at ~0.66. The goal of this implementation plan is to introduce stronger regularization techniques, soft targets, and partial backbone freezing to improve generalization and push the validation Macro-F1 higher.

## User Review Required

> [!IMPORTANT]
> The text augmentation (EDA) requires adding logic to the `MVSADataset` to randomly drop or swap words during training. Do you have a preference for any specific text augmentation library (like `nlpaug` or `nltk`), or should I implement a lightweight custom Python function to avoid adding new dependencies?

## Open Questions

> [!NOTE]
> 1. Have you inspected the confusion matrix for the last run? If the model is predicting too many "Neutral" false positives, we should lower `HP.NEUTRAL_BOOST` back to 1.0 or 1.5. 
> 2. Should we also consider reducing the `HP.BATCH_SIZE` from 32 to 16? Smaller batch sizes can introduce more gradient noise, which acts as an additional regularizer.

## Proposed Changes

### 1. Configuration & Hyperparameters (HP)
#### [MODIFY] [claude_msa_edl_co.ipynb](file:///e:/Thesis%20Claude/claude_msa_edl_co.ipynb)
- **Weight Decay**: Increase `HP.WEIGHT_DECAY` from `0.02` to `0.05` to force smaller weights.
- **Backbone Learning Rate**: Decrease `HP.BACKBONE_LR` from `5e-6` to `1e-6` to minimize catastrophic forgetting and overfitting in the large feature extractors.
- **KL Annealing**: Increase `HP.KL_ANNEALING_EPOCHS` from `5` to `10` or `15`. This gives the model more time to explore the evidence space before the KL divergence penalty heavily kicks in.

### 2. Label Smoothing & Stricter R-Drop
#### [MODIFY] [claude_msa_edl_co.ipynb](file:///e:/Thesis%20Claude/claude_msa_edl_co.ipynb)
- **Label Smoothing**: In the training loop, apply label smoothing to `y_onehot` (e.g., using `epsilon = 0.1`). Soft targets prevent the EDL model from becoming overly confident and pushing Dirichlet evidence toward infinity.
- **R-Drop**: Increase the R-Drop consistency penalty in the `total_loss` function from `0.5` to `1.5` or `2.0`. This will heavily penalize the model if the outputs differ between dropout passes, acting as a strong combatant against memorization.

### 3. Partial Backbone Freezing
#### [MODIFY] [claude_msa_edl_co.ipynb](file:///e:/Thesis%20Claude/claude_msa_edl_co.ipynb)
- Currently, the backbones are fully unfrozen at epoch 8. Fine-tuning the entirety of ResNet and BERT on 4,500 samples is a primary cause of overfitting.
- **Action**: Update the `set_backbone_grad` function to **permanently freeze** the lower layers. We will only unfreeze the last block of ResNet (`layer4`) and the last 2 layers of BERT (layers 10 and 11). 

### 4. Text Data Augmentation (EDA)
#### [MODIFY] [claude_msa_edl_co.ipynb](file:///e:/Thesis%20Claude/claude_msa_edl_co.ipynb)
- You have Mixup for images, but no augmentation for text. 
- **Action**: Introduce a `random_deletion` or `random_swap` function within the `MVSADataset` `__getitem__` method (applied only to the training set) with a 10-15% probability. This makes the text representations more robust.

## Verification Plan

### Automated Tests
- Run the training loop in the Jupyter notebook after applying the changes.
- Monitor the gap between Train F1 and Val F1. We expect Train F1 to drop slightly (e.g., to ~0.85-0.90) but Val F1 to rise (e.g., > 0.70).

### Manual Verification
- Review the output confusion matrix to verify that the neutral class is still being predicted correctly without a disproportionate amount of false positives.
