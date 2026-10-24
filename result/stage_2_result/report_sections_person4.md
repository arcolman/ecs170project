# Stage 2 report: Experiment analysis, Discussion, Conclusion (Person 4)

Numbers come from `result/stage_2_result/ablation/ablation_table.csv`, `ablation/runs.jsonl` and `selection/selection.csv`.
Test accuracy is mean ± std over seeds 1, 2, 3. Table and figure names are in [brackets].

---

## Experiment Analysis

[Table: `ablation/ablation_table.tex`] [Figures: `ablation/curves_<group>.png`]

We trained 23 MLP configurations, each with three seeds. Every configuration changes one setting of the baseline (784-256-128-10, ReLU, Adam 1e-3, batch 128, cross-entropy, 20 epochs, test accuracy 97.86 ± 0.21%), and every run is logged in `runs.jsonl`. Because the baseline's seed-to-seed spread is about 0.2%, we treat only differences larger than roughly 0.4% as reliable.

**Regularization is the one large effect.** The baseline fits the training set almost perfectly (99.72%) but reaches only 97.86% on the test set. The settings that attack this gap help most: label smoothing 0.1 gives 98.55 ± 0.02% (+0.69%) and dropout 0.2 gives 98.27 ± 0.06% (+0.41%). Both also make runs far more consistent across seeds than the baseline. Dropout 0.5 (98.09%) and weight decay (98.01%) are within noise of the baseline.

**Optimizer and learning rate mostly reflect the 20-epoch budget.** Plain SGD at lr 0.01 reaches only 93.91%, and its training loss is still falling at epoch 20, so it had not converged. Adding momentum gives 97.95%, and RMSprop gives 98.17 ± 0.03%, the best optimizer and the most consistent one. For Adam, lr 1e-3 beat 1e-4 (97.31%, too slow in 20 epochs) and 1e-2 (97.48%, its training loss hovers around 0.05 from epoch 10 on, versus about 0.01 for lr 1e-3). Each optimizer was tried at a single learning rate, so these results do not rank the optimizers in general.

**Architecture changes did not help reliably.** A network that is too small loses accuracy (64-32: 97.14%, −0.72%). Beyond that, one hidden layer (98.04%), three layers (98.16%), four layers (97.89%) and 1024-512 (97.93%) are all within about two standard deviations of the baseline. Making the network wider only cost time: 1024-512 trained 4× slower.

**Other settings.** Batch size 2048 lowers accuracy to 97.55% because it makes only about 600 updates in 20 epochs, whereas sizes 32 to 512 are equivalent (97.9 to 98.0%) and 512 trains twice as fast. Tanh, Sigmoid, MSE loss, and the Kaiming and Xavier initializations all fall within noise of the baseline.

**Model selection.** From the changes that helped, we built five candidate models and compared them on a validation split of 10,000 training images (test set unused). The winner was 512-256-128 with label smoothing 0.1, dropout 0.2 and a cosine learning-rate schedule for 30 epochs (validation 98.60 ± 0.03%). Retrained on all 60,000 images, it scores 98.67 ± 0.04% on the test set. Label smoothing alone already reaches 98.55%, so the other changes together add only about 0.1%.

---

## Discussion

The main finding is that overfitting, not model capacity, limits this MLP. Training accuracy is near 100% in almost every configuration while test accuracy stays at 97 to 98.7%, and the changes that narrow that gap (label smoothing, dropout) beat every architecture change we tried. The gains also overlap: label smoothing, dropout and a third layer give +0.69%, +0.41% and +0.30% separately, but only +0.81% together, because all three reduce overfitting. This is why the final model is only 0.12% better than label smoothing alone.

The final MLP's error rate is 1.3%, compared with 12.5% for the decision tree reference (87.50%). An MLP flattens the image into 784 numbers and ignores the 2-D layout of the pixels, which may explain why further tuning gives diminishing returns.

**Limitations.**
- Three seeds give noisy standard deviations, and several of the differences we discuss are within that noise. The reported std values are population standard deviations; sample values are about 20% larger.
- The ablation was scored on the test set, as the assignment asks, and the candidate models were built from its best results. Only the choice among the candidates used validation data, so the final test score is slightly optimistic.
- Each optimizer used one learning rate, and training lasted 20 epochs, which penalizes slow configurations such as SGD and lr 1e-4.
- The cosine schedule and the 30 epochs were never tested on their own, so we cannot say how much of the final gain they cause.
- The weight-decay run also changes the optimizer from Adam to AdamW.
- The learning-curve figures show training loss for seed 1 only. Losses of different loss functions are on different scales and cannot be compared across curves (label smoothing cannot go below about 0.5).

---

## Conclusion

Starting from a 97.86% baseline, one-change-at-a-time experiments showed that regularization is the effective lever: label smoothing alone raised test accuracy to 98.55%. The best configuration, a 512-256-128 MLP with ReLU, dropout 0.2, label smoothing 0.1, Adam with a cosine learning-rate schedule and batch size 128 for 30 epochs, was chosen on a held-out validation split and achieves 98.67 ± 0.04% test accuracy (macro F1 0.9867), far above the decision tree's 87.50%. Depth, width, activation and initialization made no reliable difference, and optimizer and learning-rate differences mainly reflected how fast each setting converges. Future work could use more seeds, tune the learning rate separately for each optimizer, and use models that exploit the image structure, such as convolutional networks.
