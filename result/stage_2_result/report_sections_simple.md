# Stage 2 report: shorter draft for Sections 2, 3.2, 3.5, 3.6, 3.7

All numbers come from the runs in `result/stage_2_result/`. Figures are named in [brackets].

---

## Section 2: Model Description

[Figure: `mlp_architecture.png`]

Our model is a fully connected network (MLP) with three hidden layers, written in PyTorch as `Method_MLP`.

- **Input (784):** one 28 x 28 image, flattened, with pixel values scaled to [0, 1].
- **Hidden layers (512, 256, 128):** each is a linear layer, then ReLU, then dropout with p = 0.2. Dropout randomly turns off 20% of the units during training so the network can't lean on any single unit. It is off at test time.
- **Output (10):** one score (logit) per digit. We don't apply softmax in the model, because PyTorch's `CrossEntropyLoss` already does it.
- **Prediction:** the class with the highest score.

The network has 567,434 trainable parameters. For each mini-batch we compute the logits, compute the loss, call `loss.backward()` to get the gradients by back-propagation, and let the Adam optimizer update the weights.

We use cross-entropy with label smoothing 0.1: the true class gets target 0.91 andP each other class gets 0.01. This stops the model from becoming over-confident on the training images, which is where most of our test error came from (Section 3.7).

---

## Section 3.2: Detailed Experimental Setups

**Final model**

| Setting | Value |
|---|---|
| Architecture | 784 - 512 - 256 - 128 - 10 |
| Activation | ReLU on hidden layers |
| Dropout | 0.2 after each hidden layer |
| Loss | Cross-entropy, label smoothing 0.1 |
| Optimizer | Adam, lr 1e-3, no weight decay |
| LR schedule | Cosine decay from 1e-3 to 0 |
| Batch size | 128, shuffled each epoch |
| Epochs | 30 |
| Initialization | PyTorch default for `nn.Linear` |
| Seeds | 1, 2, 3 (we report mean and std) |
| Hardware | CPU, about 72 s per run |

**Baseline (reference for the ablation):** 784 - 256 - 128 - 10, ReLU, no dropout, plain cross-entropy, Adam with lr 1e-3, batch size 128, 20 epochs, 235,146 parameters.

**How we picked the final model.** We first changed one baseline setting at a time (Section 3.7). Then we combined the changes that helped into five candidate models. To avoid choosing by test score, we held out 10,000 of the 60,000 training images as a validation set, trained each candidate on the other 50,000, and kept the best one (validation accuracy 98.60% +/- 0.03%). We then retrained it on all 60,000 images and tested it once on the test set. The runner-up (1024-512, dropout 0.3) got 98.58%, and the simpler candidates got 98.2% to 98.4%.

**GPU (optional step 2-6).** The code accepts `device='cuda'` or `'mps'`. In a quick test on a Mac, MPS was about 4x slower than CPU for this small network (6.6 s vs 1.6 s for 3 epochs), so we trained on CPU.

---

## Section 3.5: Training Convergence Plot

[Figure: `mlp_final_learning_curve.png`]

The plot shows the mean training loss for each of the 30 epochs. The loss falls fast in the first 3 epochs (0.79 to 0.58), keeps decreasing, and flattens after about epoch 20, so training converges. It levels off near 0.51 instead of 0 because of label smoothing: with smoothed targets, the lowest possible loss is about 0.50, so the model is almost at that floor. Training accuracy ends at 99.99%.

---

## Section 3.6: Model Performance

Test results on the 10,000 test images (mean +/- std over seeds 1, 2, 3):

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | F1 (macro) |
|---|---|---|---|---|---|
| Decision tree (reference) | 0.8750 | 0.8751 | 0.8750 | 0.8750 | 0.8735 |
| MLP baseline (256-128) | 0.9786 +/- 0.0021 | 0.9788 | 0.9786 | 0.9786 | 0.9784 |
| **MLP final (512-256-128)** | **0.9867 +/- 0.0004** | **0.9867** | **0.9867** | **0.9867** | **0.9867** |

The saved predictions come from seed 1 alone: accuracy 0.9862, macro F1 0.9861. Re-running on another machine can shift the third decimal.

The MLP cuts the error rate from 12.5% (decision tree) to 1.3%. The classes are nearly balanced, so weighted and macro scores differ by at most 0.0001. Per-class F1 ranges from 0.977 (digit 9) to 0.994 (digit 1). The most common mistakes are between similar-looking digits: 4 predicted as 9 (11 times) and 9 as 4 (8 times). Together, 4 and 9 account for 19 of the 138 errors.

---

## Section 3.7: Ablation Studies

[Table: `ablation/ablation_table.tex`] [Figures: `ablation/curves_<group>.png`]

Starting from the baseline, we changed one setting at a time and trained each version with three seeds. Test accuracy, mean +/- std:

| Changed setting | Value | Test accuracy (%) |
|---|---|---|
| none | **baseline** | 97.86 +/- 0.21 |
| Depth | 1 hidden layer (256) | 98.04 +/- 0.09 |
| | 3 hidden layers (512-256-128) | 98.16 +/- 0.24 |
| | 4 hidden layers | 97.89 +/- 0.19 |
| Width | 64-32 | 97.14 +/- 0.13 |
| | 1024-512 | 97.93 +/- 0.22 |
| Activation | Tanh | 98.10 +/- 0.13 |
| | Sigmoid | 97.96 +/- 0.13 |
| Optimizer | SGD (lr 0.01) | 93.91 +/- 0.08 |
| | SGD + momentum | 97.95 +/- 0.05 |
| | RMSprop | 98.17 +/- 0.03 |
| Learning rate (Adam) | 1e-4 | 97.31 +/- 0.10 |
| | 1e-2 | 97.48 +/- 0.08 |
| Loss | MSE | 98.07 +/- 0.10 |
| | Label smoothing 0.1 | **98.55 +/- 0.02** |
| Batch size | 32 | 97.92 +/- 0.21 |
| | 512 | 97.98 +/- 0.03 |
| | 2048 | 97.55 +/- 0.06 |
| Regularization | Dropout 0.2 | 98.27 +/- 0.06 |
| | Dropout 0.5 | 98.09 +/- 0.12 |
| | Weight decay 1e-4 (AdamW) | 98.01 +/- 0.13 |
| Initialization | Kaiming normal | 97.99 +/- 0.08 |
| | Xavier uniform | 97.84 +/- 0.24 |

Because the baseline varies by about 0.2% between seeds, we only treat gaps larger than roughly 0.4% as real.

**What we found**

- **Overfitting is the main problem, not model size.** The baseline gets 99.7% on training data but 97.9% on test data. The two settings that fight this help most: label smoothing (+0.69%) and dropout 0.2 (+0.41%).
- **Size matters little, except when it is too small.** The 64-32 network loses 0.7%. A third layer, a fourth layer or a wider network all fall within noise of the baseline, and the 1024-512 network trains about 4x slower.
- **Optimizer and learning-rate gaps mostly come from the 20-epoch limit.** Plain SGD was still improving at epoch 20, so it hadn't converged. Lr 1e-4 is too slow. Lr 1e-2 is noisy: its training loss hovers around 0.05 from epoch 10 on, compared with about 0.01 for 1e-3. We tried each optimizer at only one learning rate, so this doesn't rank them in general.
- **Very large batches hurt.** Batch 2048 makes only about 600 updates in 20 epochs, and accuracy drops to 97.55%. Sizes 32 to 512 are about the same.
- **Activation, initialization and MSE loss** all landed within noise of the baseline.
- **The gains overlap.** Label smoothing, dropout and the extra layer each help, but together with the cosine schedule they reach 98.67% (+0.81%), only 0.12% above label smoothing alone. They all reduce overfitting, so they don't add up.

Note: the ablation scores are test-set scores, as the template asks. The final model was chosen on the validation split, not on these numbers, but the candidates were built from the ablation's best results, so the final test score may be slightly optimistic.
