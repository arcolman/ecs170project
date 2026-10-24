# Stage 2 report: draft text for Sections 2, 3.2, 3.5, 3.6, 3.7

All numbers come from the runs in `result/stage_2_result/`. Figures to insert are named in [brackets].

---

## Section 2: Model Description

[Figure: `result/stage_2_result/mlp_architecture.png`]

Our model is a fully connected feed-forward network (Multi-Layer Perceptron) with three hidden layers, implemented in PyTorch as `Method_MLP` (inheriting from the template's `method` class and `nn.Module`).

- **Input layer (784):** the 28 × 28 pixel intensities of one digit image, flattened and scaled to [0, 1].
- **Hidden layers (512 → 256 → 128):** each is a linear layer `h = W x + b` followed by a ReLU activation `max(0, h)` and dropout with p = 0.2. The layers get narrower so the network compresses the pixel input into progressively more abstract features. Dropout randomly zeroes 20% of the hidden units at each training step, which keeps the network from relying on any single unit and reduces overfitting. Dropout is switched off at test time.
- **Output layer (10):** a linear layer producing one unnormalized score (logit) per digit class. We do not apply Softmax inside the model, because PyTorch's `CrossEntropyLoss` applies log-softmax itself and is more numerically stable that way.
- **Prediction:** the class with the largest logit, which is the same as the class with the largest softmax probability.

The network has 567,434 trainable parameters. It is trained with mini-batch gradient descent: the forward pass computes the logits for a batch, the cross-entropy loss compares them with the true labels, `loss.backward()` computes the gradient of the loss with respect to every weight by back-propagation, and the Adam optimizer updates the weights.

**Loss function.** We use cross-entropy with label smoothing 0.1. Instead of a one-hot target, the true class gets probability 0.91 and every other class gets 0.01. This penalizes the model for becoming over-confident on the training images, which in our experiments was the main cause of the gap between training accuracy (≈100%) and test accuracy (Section 3.7).

---

## Section 3.2: Detailed Experimental Setups

**Final model**

| Setting | Value |
|---|---|
| Architecture | 784 → 512 → 256 → 128 → 10 (3 hidden layers) |
| Activation | ReLU (hidden layers), none on the output (logits) |
| Dropout | 0.2 after each hidden activation |
| Loss | Cross-entropy with label smoothing 0.1 |
| Optimizer | Adam, initial learning rate 1e-3, no weight decay |
| Learning-rate schedule | Cosine annealing from 1e-3 to 0 over the 30 epochs |
| Batch size | 128, reshuffled every epoch |
| Epochs | 30 (≈ 14,100 gradient updates) |
| Initialization | PyTorch default for `nn.Linear` (Kaiming-uniform weights, uniform biases in ±1/√fan_in) |
| Random seeds | 1, 2, 3 (results reported as mean ± std over the three) |
| Hardware | CPU (Apple M-series Mac), ≈ 72 s training time per run |

**Baseline model** (the reference point for the ablation study): 784 → 256 → 128 → 10, ReLU, no dropout, plain cross-entropy, Adam with lr 1e-3 (constant), batch size 128, 20 epochs, 235,146 parameters.

**How we chose the final model.** First, in the ablation study (Section 3.7), we changed one setting of the baseline at a time. We then combined the changes that helped most (label smoothing, dropout, a third hidden layer) into five candidate models. To avoid choosing the final model by its test score, we held out a stratified 10,000 of the 60,000 training images as a validation set, trained each candidate on the remaining 50,000 (2 seeds each), and picked the one with the best validation accuracy (98.60% ± 0.03%). That model was then retrained on all 60,000 training images and evaluated on the test set once.

| Candidate (all Adam, batch 128) | Validation accuracy |
|---|---|
| **512-256-128, label smoothing, dropout 0.2, cosine LR, 30 epochs** | **98.60% ± 0.03%** |
| 1024-512, label smoothing, dropout 0.3, cosine LR, 30 epochs | 98.58% ± 0.02% |
| 512-256-128, label smoothing, dropout 0.2, 20 epochs | 98.36% ± 0.15% |
| 256-128, label smoothing, 20 epochs | 98.34% ± 0.05% |
| 256-128, label smoothing, dropout 0.2, 20 epochs | 98.24% ± 0.04% |

**GPU (optional step 2-6).** The code runs on CUDA or Apple MPS with `device='cuda'` / `device='mps'`. We tested MPS on our Mac, and for this small network it was about 4× slower than the CPU (6.6 s vs 1.6 s for 3 epochs), because launching thousands of tiny matrix multiplications on the GPU costs more than the computation itself. We therefore trained on the CPU.

---

## Section 3.5: Training Convergence Plot

[Figure: `result/stage_2_result/mlp_final_learning_curve.png`]

The plot shows the final model's mean training loss in each of the 30 epochs. The loss drops quickly in the first 3 epochs (0.79 → 0.58), then decreases steadily and flattens out after about epoch 20, so training converges. The curve flattens at about 0.51 rather than at 0 because of label smoothing. With smoothed targets, the lowest loss any model can reach is the entropy of the target distribution, −(0.91 ln 0.91 + 9 × 0.01 ln 0.01) ≈ 0.50, so the model ends very close to this minimum. The cosine learning-rate schedule makes the curve smooth at the end: the steps get smaller as the learning rate approaches 0. Training accuracy reaches 99.99%.

[Optional figure: `result/stage_2_result/ablation/curves_optimizer.png` shows the convergence of the four optimizers on a log scale. Adam and RMSprop converge fastest, plain SGD is much slower.]

---

## Section 3.6: Model Performance

Final model on the 10,000 test images (mean ± std over seeds 1, 2, 3):

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | F1 (macro) |
|---|---|---|---|---|---|
| Decision tree (reference) | 0.8750 | 0.8751 | 0.8750 | 0.8750 | 0.8735 |
| MLP baseline (256-128) | 0.9786 ± 0.0021 | 0.9788 | 0.9786 | 0.9786 | 0.9784 |
| **MLP final (512-256-128)** | **0.9867 ± 0.0004** | **0.9867** | **0.9867** | **0.9867** | **0.9867** |

Seed 1 alone, the run whose predictions are saved: accuracy 0.9862, weighted P/R/F1 0.9862, macro P/R/F1 0.9861.

The MLP cuts the decision tree's error rate from 12.5% to 1.3%. Because the classes are nearly balanced, the weighted and macro averages differ by at most 0.0001, and per-class F1 is between 0.977 (digit 9) and 0.994 (digit 1). The most common mistakes are between digits that look alike: 4 predicted as 9 (11 times) and 9 predicted as 4 (8 times), then 7 ↔ 9, 8 → 9 and 7 → 2. Together, 4 ↔ 9 accounts for 19 of the 138 test errors.

---

## Section 3.7: Ablation Studies

[Table: `result/stage_2_result/ablation/ablation_table.tex` (full) or the condensed table below]
[Figures: `result/stage_2_result/ablation/curves_<group>.png`]

Starting from the baseline (Section 3.2), we changed one setting at a time and kept everything else fixed. Each configuration was trained with three seeds. Test accuracy, mean ± std:

| Changed setting | Value | Test accuracy (%) |
|---|---|---|
| — | **baseline** (256-128, ReLU, Adam 1e-3, batch 128, CE, no dropout) | 97.86 ± 0.21 |
| Depth | 1 hidden layer (256) | 98.04 ± 0.09 |
| | 3 hidden layers (512-256-128) | 98.16 ± 0.24 |
| | 4 hidden layers (512-256-128-64) | 97.89 ± 0.19 |
| Width | 64-32 | 97.14 ± 0.13 |
| | 1024-512 | 97.93 ± 0.22 |
| Activation | Tanh | 98.10 ± 0.13 |
| | Sigmoid | 97.96 ± 0.13 |
| Optimizer | SGD (lr 0.01) | 93.91 ± 0.08 |
| | SGD + momentum 0.9 (lr 0.01) | 97.95 ± 0.05 |
| | RMSprop (lr 1e-3) | 98.17 ± 0.03 |
| Learning rate (Adam) | 1e-4 | 97.31 ± 0.10 |
| | 1e-2 | 97.48 ± 0.08 |
| Loss | MSE (softmax output vs one-hot) | 98.07 ± 0.10 |
| | Cross-entropy + label smoothing 0.1 | **98.55 ± 0.02** |
| Batch size | 32 | 97.92 ± 0.21 |
| | 512 | 97.98 ± 0.03 |
| | 2048 | 97.55 ± 0.06 |
| Regularization | Dropout 0.2 | 98.27 ± 0.06 |
| | Dropout 0.5 | 98.09 ± 0.12 |
| | Weight decay 1e-4 (AdamW) | 98.01 ± 0.13 |
| Initialization | Kaiming normal | 97.99 ± 0.08 |
| | Xavier uniform | 97.84 ± 0.24 |

**Observations**

- **Overfitting is the main limitation, not model capacity.** The baseline reaches 99.7% training accuracy but only 97.9% test accuracy, and its test accuracy varies by ±0.2% between seeds. The changes that help most are the ones that regularize: label smoothing (+0.69%) and dropout 0.2 (+0.41%). Both also make results much more consistent across seeds (std 0.02–0.06% vs 0.21%).
- **Depth and width help only up to a point.** Going from 2 to 3 hidden layers gives +0.30%, but a 4th layer gives no gain, and doubling the width to 1024-512 gives almost nothing (+0.07%) while making training 4× slower. A network that is too small (64-32) loses 0.7%.
- **The optimizer matters most for convergence speed.** Plain SGD with lr 0.01 is far behind after 20 epochs (93.9%). Its training loss is still falling in the curves, so it simply has not converged yet. Adding momentum closes almost the whole gap, and RMSprop is as good as or slightly better than Adam.
- **The learning rate has a sweet spot around 1e-3 for Adam.** 1e-4 converges too slowly within 20 epochs, and 1e-2 is too large: its loss stalls around 0.05 and goes up and down from epoch to epoch, while 1e-3 keeps decreasing to below 0.01.
- **Very large batches hurt.** With batch size 2048 there are only 600 updates in 20 epochs (vs ~9,400 with batch 128) and accuracy drops to 97.55%. Batch sizes from 32 to 512 perform about the same, and 512 trains 2× faster than 128.
- **Activation and initialization make small differences** (within about two standard deviations of the baseline). Tanh and sigmoid did slightly better than ReLU at this depth, though the difference is close to the seed-to-seed noise.
- **MSE loss works but trains differently.** Its loss values are on a different scale from cross-entropy, so the curves cannot be compared directly, but it reaches 98.07%.
- **Combining the best changes adds up.** Label smoothing, dropout, 3 hidden layers and a cosine learning-rate schedule together reach 98.67% ± 0.04% (Section 3.6), 0.81% above the baseline.

Note: The ablation scores are test-set scores, as the report template asks for. The final configuration, however, was chosen on a held-out validation split of the training data (Section 3.2), not on these test scores.
