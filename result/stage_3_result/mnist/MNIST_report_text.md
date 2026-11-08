# MNIST report text — Aron Colman

Paste the following paragraphs into the matching sections of the team report. This covers MNIST only; combine it with the ORL and CIFAR contributions within the five-page limit.

## Section 2: Model description (MNIST)

Our MNIST CNN maps a 1x28x28 grayscale image to 10 class logits. The first block applies a 5x5 convolution with 16 filters and padding 2, ReLU, and 2x2 max pooling, producing a 16x14x14 feature map. The second block applies a 5x5 convolution with 32 filters and padding 2, ReLU, and 2x2 max pooling, producing a 32x7x7 feature map. Flattening gives 1,568 features, which feed a linear layer with 10 outputs. The model has 28,938 trainable parameters. Cross-entropy receives the raw logits; predictions use the largest logit.

Figure: `mnist_architecture.png`.

## Section 3.2: Experimental settings (MNIST)

We used the instructor's predefined 60,000/10,000 training/test split, converted pixels to float32 in [0,1], and kept labels 0-9. All four configurations used PyTorch's default layer initialization, random seed 1, Adam with learning rate 0.001, batch size 128, cross-entropy loss, and 10 epochs. We used no dropout, batch normalization, augmentation, weight decay, or learning-rate schedule. Runs used PyTorch 2.15.0+cpu on Windows with four CPU threads. The baseline training loop took 264.6 seconds, excluding final evaluation and saving.

## Section 3.5: Training convergence (MNIST)

Baseline training loss decreased from 0.2692 at epoch 1 to 0.0149 at epoch 10. All four configurations showed decreasing loss over training. The final baseline training accuracy, evaluated with the final weights, was 99.55%; test accuracy was 98.91%. The loss curves show training convergence but do not establish validation convergence.

Figures: `baseline_learning_curve.png` for the main model; `ablation_learning_curves.png` for the comparison.

## Section 3.6: Model performance and Section 3.7: Ablation studies (MNIST)

| Configuration | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) |
|---|---:|---:|---:|---:|---:|
| Baseline: two blocks, 5x5, max pool | 98.91% | 98.92% | 98.91% | 98.91% | 98.91% |
| 3x3 kernels | 98.88% | 98.88% | 98.88% | 98.88% | 98.87% |
| One conv block | 98.50% | 98.51% | 98.50% | 98.50% | 98.49% |
| Average pooling | 98.86% | 98.87% | 98.86% | 98.86% | 98.85% |

Precision, recall and F1 marked (w) are weighted by class support. Macro F1 weights every digit class equally. Values are percentages.

Each ablation changed one architecture setting relative to the baseline. Switching both convolution kernels from 5x5 to 3x3 gave 98.88% accuracy (-0.03 percentage points versus baseline). Removing the second convolution/pooling block gave 98.50% (-0.41 points). The one-block model therefore performed worse in this run, suggesting that the second block helped recognition under these training settings. The removed pooling stage also increases the flattened feature size to 3,136, so this depth ablation changes the classifier size as well. Replacing both max-pooling operations with average pooling gave 98.86% (-0.05 points).

These are single-seed comparisons with a fixed epoch budget; they do not measure run-to-run uncertainty or establish that small differences are reliable. The baseline was retained as the predefined reference model. The test set was used for reporting these architecture comparisons, not for validation-based model selection.
