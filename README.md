# ECS 170 Course Project (Fall 2026)

Team course project for ECS 170 (Artificial Intelligence) at UC Davis. Each stage builds a classic AI model in PyTorch on top of the instructor's code template, evaluates it on the provided train/test split, and writes up the results in a report of 5 pages or less.

| Stage | Task | Status |
|---|---|---|
| 1 | Group formation and environment setup | Done |
| 2 | Handwritten digit classification with a multi-layer perceptron (MLP) | Done |
| 3 | Image classification with convolutional neural networks (CNN) on MNIST, ORL faces and CIFAR-10 | In progress |
| 4 | Text classification and generation with recurrent neural networks (RNN) | Not started |
| 5 | Node classification with graph neural networks (GCN) | Not started |

## Setup

Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Every script runs as a module from the project root, e.g. `.venv/bin/python -m script.stage_3_script.check_data`. On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.

The datasets are too large for GitHub and are git-ignored. Get them from the course page (or the group drive) and place them as described for each stage below.

## Repository layout

```
local_code/
  base_class/         template base classes: dataset, method, result, setting, evaluate
  stage_<n>_code/     concrete classes for stage n (loaders, models, metrics, report helpers)
script/
  stage_<n>_script/   runnable scripts and tests for stage n
data/
  stage_<n>_data/     instructor datasets (git-ignored)
result/
  stage_<n>_result/   saved predictions, metrics tables and plots used in the reports
```

Every experiment follows the template pipeline:
`Dataset_Loader` → `Method_*` → `Result_Saver` → `Evaluate_Metrics`, wired together by `Setting_Predefined_Split`, which trains on the given training set and tests once on the given test set, without re-splitting. `Evaluate_Metrics` reports accuracy plus weighted, macro and micro precision, recall and F1, since the tasks are multiclass.

## Stage 2: MLP

Put the instructor's `train.csv` and `test.csv` in `data/stage_2_data/`, then:

```bash
.venv/bin/python -m script.stage_2_script.check_data            # verify the data loads correctly
.venv/bin/python -m script.stage_2_script.script_mlp            # final model: scores, learning curve, saved predictions
.venv/bin/python -m script.stage_2_script.script_mlp_ablation   # every ablation config x 3 seeds (~20 min on CPU)
.venv/bin/python -m script.stage_2_script.script_mlp_select     # pick the final config on a validation split
.venv/bin/python -m script.stage_2_script.script_decision_tree  # non-neural reference point
.venv/bin/python -m script.stage_2_script.plot_architecture     # architecture diagram for the report
```

- `Method_MLP` takes every setting as a keyword (`hidden_dims`, `activation`, `dropout`, `optimizer_name`, `learning_rate`, `batch_size`, `max_epoch`, `loss_name`, `init_name`, ...). The configs live in `script/stage_2_script/mlp_configs.py`: `BASELINE`, the one-change-at-a-time `ABLATIONS`, and `FINAL`.
- The ablation script resumes where it stopped if interrupted. `--fresh` starts over, and group names (e.g. `depth optimizer`) run only those groups.
- **Result:** the final MLP (784 → 512 → 256 → 128 → 10, dropout 0.2, label smoothing 0.1) reaches **98.62%** test accuracy, compared with 87.50% for the decision tree. The full table is in `result/stage_2_result/results_table.csv`, and draft report text is in `result/stage_2_result/report_sections_draft.md`.

## Stage 3: CNN

Unzip the instructor's `stage_3_data.zip` so that the `MNIST`, `ORL` and `CIFAR` files sit directly in `data/stage_3_data/`, then:

```bash
.venv/bin/python -m script.stage_3_script.check_data        # shapes, label ranges and a sample image per dataset
.venv/bin/python -m script.stage_3_script.script_cnn_mnist  # train and test the MNIST CNN
.venv/bin/python -m script.stage_3_script.script_cnn_mnist_ablation  # three MNIST ablations, resume completed runs
.venv/bin/python -m script.stage_3_script.test_cnn          # quick check on synthetic data (no datasets needed)
```

| Dataset | Images | Train / test | Classes | Loader |
|---|---|---|---|---|
| MNIST | 1 × 28 × 28 grayscale digits | 60,000 / 10,000 | 10 | `Dataset_Loader_MNIST` |
| ORL | 1 × 112 × 92 grayscale faces | 360 / 40 | 40 | `Dataset_Loader_ORL` |
| CIFAR-10 | 3 × 32 × 32 color objects | 50,000 / 10,000 | 10 | `Dataset_Loader_CIFAR` |

- The loaders return `(N, C, H, W)` float32 pixels scaled to [0, 1] and int64 labels starting at 0. ORL is stored as three identical RGB channels, so its loader keeps one channel and shifts its labels from 1–40 to 0–39.
- `Method_CNN_MNIST` takes its architecture as settings: `conv_channels`, `kernel_size`, `padding`, `stride`, `pool` (`'max'` or `'avg'`), `batch_norm`, `fc_dims`, `dropout`, plus the same training options as the MLP. It uses an NVIDIA GPU (CUDA) or an Apple GPU (MPS) when one is available.
- **Config experiments:** in `script_cnn_mnist.py`, set `run_name` and `settings` (e.g. `settings = {'kernel_size': 3}`) and run the script again. Each run is added to `result/stage_3_result/mnist/results_table.csv`, along with its own learning curve.
- **MNIST ablations:** `script_cnn_mnist_ablation` runs 3x3 kernels, one convolution block, and average pooling, each changing one baseline setting. It uses seed 1, CPU with four threads, and 10 epochs. Run the baseline script first. Each experiment saves its own predictions, weights, history, summary and curve; the script also generates combined loss curves and a metric comparison. An existing `<run>_summary.json` marks a completed run. These test-set comparisons describe ablations; they do not perform validation-based model selection.
- **ORL / CIFAR models:** copy `Method_CNN_MNIST.py` and the MNIST script, then change `in_channels`, `input_size` and `output_dim`. Save their results to `result/stage_3_result/orl/` and `result/stage_3_result/cifar/`.
- **Baseline result:** 2 conv blocks (16 and 32 filters, 5 × 5 kernels, ReLU, max-pool) and one fully connected layer reach **98.94%** test accuracy on MNIST (weighted F1 0.9894) after 10 epochs, taking 33 s on an Apple M-chip.
- **Aron's MNIST reproduction and ablations:** the Windows CPU baseline reaches **98.91%** accuracy; 3x3 kernels reach **98.88%**, one conv block **98.50%**, and average pooling **98.86%**. All use seed 1 and 10 epochs. The current MNIST table, baseline history and plots use this CPU reproduction; the JSON-lines log also retains the earlier Apple run. Draft report text is in `result/stage_3_result/mnist/MNIST_report_text.md`.

## Contributing

Work on a branch, open a pull request into `main`, and get a teammate to look it over before merging. Commit code and the result files used in the report, but not the datasets.
