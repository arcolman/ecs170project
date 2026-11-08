# Stage 3 report: ORL draft text (Sections 2, 3.2, 3.5, 3.6, 3.7)

All numbers come from the runs in `result/stage_3_result/orl/`. Figures to insert are named in [brackets].
Test scores are the mean ± standard deviation over 3 random seeds (1, 2, 3), because the ORL test set has only 40
images: one misclassified face moves accuracy by 2.5 percentage points.

---

## Section 2: Model Description (ORL)

[Figure: `result/stage_3_result/orl/final_architecture.png`]

The ORL model (`Method_CNN_ORL`) takes a 1 × 112 × 92 grayscale face and outputs 40 logits, one per person. It
starts from our MNIST baseline and differs in three ways, all aimed at the larger images and the very small
training set (9 images per person):

- **Three conv blocks instead of two (16 → 32 → 64 filters, 5 × 5 kernels, "same" padding).** Each block is
  Conv2d → ReLU → 2 × 2 max-pooling, so the feature map shrinks from 112 × 92 to 56 × 46, 28 × 23 and finally
  14 × 11. Pooling three times keeps the flattened vector at 64 × 14 × 11 = 9,856 values; with only two blocks it would
  be 20,608.
- **A 128-unit hidden fully connected layer** (ReLU) between the flattened features and the 40-way output layer.
- **Dropout with p = 0.5** before the hidden layer and before the output layer. During training this randomly zeroes
  half of these units at each step, so the network cannot memorize the 360 training faces through a few specific
  units. Dropout is switched off at test time.

The model has 1,331,368 trainable parameters. The output layer has no softmax, because `CrossEntropyLoss` applies
log-softmax itself; the predicted person is the class with the largest logit.

---

## Section 3.2: Detailed Experimental Setups (ORL)

| Setting | Final model | Baseline |
|---|---|---|
| Conv blocks | 3 (16, 32, 64 filters) | 2 (16, 32 filters) |
| Kernel / padding / stride | 5 × 5 / 2 ("same") / 1 | same |
| Pooling | 2 × 2 max-pooling after every block | same |
| Fully connected layers | 9,856 → 128 → 40 | 20,608 → 40 |
| Dropout | 0.5 before each FC layer | none |
| Optimizer / loss | Adam, lr 1e-3; cross-entropy | same |
| Batch size / epochs | 32 / 40 (12 mini-batches per epoch) | same |
| Parameters | 1,331,368 | 837,608 |
| Training time | ≈ 35 s per run on an Apple M-chip GPU (MPS) | ≈ 25 s |

Compared with MNIST we use smaller mini-batches (32 instead of 128) and more epochs (40 instead of 10), because
360 training images give only 12 gradient updates per epoch. Weights use PyTorch's default initialization with a
fixed seed.

**How we chose the final model.** The ablation study (Section 3.7) changed one setting of the baseline at a time.
We then combined the changes that helped into six candidates. To avoid choosing a model by its test score, we held out
one of the nine training images of each person (40 images) as a validation set, trained each candidate on the other
320 images with 3 seeds, and kept the best one. That model was then retrained on all 360 images and tested once.

| Candidate | Validation accuracy |
|---|---|
| **3 conv blocks + FC 128 + dropout 0.5** | **99.17% ± 1.18%** |
| Baseline | 98.33% ± 1.18% |
| Baseline + dropout 0.5 | 98.33% ± 1.18% |
| 32–64 filters + 3 × 3 kernels + dropout 0.5 | 98.33% ± 1.18% |
| 3 × 3 kernels + dropout 0.5 | 97.50% ± 0.00% |
| 32–64 filters + dropout 0.5 | 97.50% ± 0.00% |

---

## Section 3.5: Training Convergence Plot (ORL)

[Figure: `result/stage_3_result/orl/final_learning_curve.png`]

The training loss stays near ln(40) ≈ 3.69 (a uniform guess over 40 people) for the first five epochs, then drops
quickly between epochs 5 and 12 as the convolutional filters learn useful face features, and levels off around 0.15
by epoch 40. The curve keeps a small amount of noise at the end because dropout changes the network at every step;
the baseline without dropout drives its training loss all the way to about 1e-4.

---

## Section 3.6: Model Performance (ORL)

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) | Train acc. |
|---|---|---|---|---|---|---|
| Final (3 conv + FC 128 + dropout 0.5) | 95.83% ± 3.12% | 0.938 ± 0.047 | 0.958 ± 0.031 | 0.944 ± 0.042 | 0.944 ± 0.042 | 100% |
| Baseline (2 conv, no dropout) | 96.67% ± 1.18% | 0.950 ± 0.018 | 0.967 ± 0.012 | 0.956 ± 0.016 | 0.956 ± 0.016 | 100% |

Per seed, the final model scored 95.0%, 92.5% and 100% (38, 37 and 40 of the 40 test faces). The test set has
exactly one image per person, so the weighted and macro averages are identical and recall equals accuracy.
Precision is a little lower than recall because one mistake costs two classes: the true person loses recall and the
wrongly predicted person loses precision.

The final model did not beat the baseline on the test set, even though it had the best validation accuracy. The
difference is about one image out of 40 and is well within the seed-to-seed spread, so the honest conclusion is that
all of our reasonable ORL CNNs score about 95–97.5%, and this small test set cannot separate them.

**Overfitting.** [Figure: `result/stage_3_result/orl/final_train_vs_test_accuracy.png`] Every model we trained
reached 100% training accuracy, while test accuracy stayed at about 95–97.5%. The gap is small because faces of
the same person in ORL are very similar (small changes in pose and expression). In the plot, training accuracy is the
running accuracy during each epoch, measured with dropout active, which is why it sits slightly *below* the test
accuracy (measured with dropout off). With dropout off, the final model classifies all 360 training images correctly.
Dropout 0.5 made the results more consistent across seeds (97.5% on all three seeds versus 95–97.5% for the baseline)
but did not raise the average.

---

## Section 3.7: Ablation Studies (ORL)

[Figures: `result/stage_3_result/orl/ablation/curves_<group>.png` — depth, dropout, filters, kernel, optimizer, pooling]

Each row changes **one** setting of the baseline; test accuracy is the mean ± std over 3 seeds, and every run reached
100% training accuracy.

| Group | Configuration | Test accuracy | F1 (w) | Params | Time |
|---|---|---|---|---|---|
| — | Baseline (16–32 filters, 5 × 5, max-pool, Adam 1e-3) | 96.67% ± 1.18% | 0.956 | 838K | 23 s |
| Dropout | 0.3 | 96.67% ± 1.18% | 0.956 | 838K | 23 s |
| | 0.5 | 97.50% ± 0.00% | 0.967 | 838K | 22 s |
| Filters | 8–16 | 96.67% ± 1.18% | 0.956 | 416K | 14 s |
| | 32–64 | 97.50% ± 0.00% | 0.967 | 1.70M | 48 s |
| Depth | 1 conv block (16) | 97.50% ± 2.04% | 0.967 | 1.65M | 16 s |
| | 3 conv blocks (16–32–64) | 96.67% ± 1.18% | 0.956 | 459K | 27 s |
| | 4 conv blocks (16–32–64–128) | 95.00% ± 2.04% | 0.938 | 449K | 35 s |
| Kernel size | 3 × 3 | 97.50% ± 0.00% | 0.967 | 829K | 21 s |
| | 7 × 7 | 96.67% ± 1.18% | 0.956 | 850K | 39 s |
| Pooling | average instead of max | 95.83% ± 1.18% | 0.944 | 838K | 23 s |
| Optimizer | Adam, lr 1e-4 | 97.50% ± 2.04% | 0.967 | 838K | 27 s |
| | SGD + momentum 0.9, lr 0.01 | 95.00% ± 0.00% | 0.933 | 838K | 26 s |

**What we learned.**
- **Every configuration lands between 95% and 97.5%**, i.e. within one test image of the others. The most useful
  changes made the results *steadier* across seeds: dropout 0.5, 3 × 3 kernels and wider 32–64 filters each gave
  97.5% on all three seeds.
- **Too deep hurts.** Four conv blocks shrink the 112 × 92 face to a 7 × 5 map, throwing away detail, and gave the
  lowest accuracy (95.0%). One block (a huge 1.65M-parameter FC layer on 56 × 46 features) was the noisiest.
- **Max pooling beats average pooling** (96.7% vs 95.8%): keeping the strongest response in each window preserves
  sharp features such as eyes and edges that averaging blurs.
- **Optimizer and learning rate change the speed of training more than the final result** (see
  `curves_optimizer.png`): Adam with lr 1e-3 converges fastest, SGD with momentum stalls for about 6 epochs before
  dropping, and Adam with lr 1e-4 still has a loss 20× higher at epoch 40, yet all three fit the training set
  perfectly.
- **Main limitation:** with 40 test images, differences under about 2.5 points are not meaningful. Repeating each
  run with 3 seeds and choosing the final model on a held-out validation split were our ways of dealing with this.
