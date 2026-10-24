# ECS 170 Stage 2 — Aron data pipeline

From the project root, install `numpy` and run:

```bash
python -m pip install numpy
python -m script.stage_2_script.check_data
```

The loader returns `{'train': {'X': ..., 'y': ...}, 'test': {'X': ..., 'y': ...}}` with float32 pixel features scaled to [0,1] and int64 labels 0–9. No re-splitting or cross-validation.

**For Kartik:** Adapt `local_code/stage_2_code/Method_MLP.py` to accept 784 input features and produce 10 logits. With `CrossEntropyLoss`, do not apply Softmax to the logits before computing loss. For large training data, use mini-batches instead of full-batch training. Wire the script to `Setting_Predefined_Split` and the Stage 2 dataset path. `Setting_Predefined_Split` follows the template's save/evaluate API. The copied Stage 1 MLP is deliberately not rewritten here because that is Kartik's workstream.

**For teammate integration:** `Dataset_Loader.load()` returns both predefined splits. Use `Setting_Predefined_Split`, not `Setting_KFold_CV` or `Setting_Train_Test_Split`. Set result folder to `result/stage_2_result/`.

## Stage 2 MLP

Setup (Python 3.10+), from the project root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put the instructor's `train.csv` and `test.csv` in `data/stage_2_data/` (they are gitignored), then:

```bash
.venv/bin/python -m script.stage_2_script.script_mlp            # final model: scores, learning curve, saved predictions
.venv/bin/python -m script.stage_2_script.script_mlp_ablation   # every ablation config x 3 seeds (~20 min on CPU)
.venv/bin/python -m script.stage_2_script.plot_architecture     # architecture diagram for Section 2
```

- `Method_MLP` takes every setting as a keyword (`hidden_dims`, `activation`, `dropout`, `optimizer_name`, `learning_rate`, `batch_size`, `max_epoch`, `loss_name`, `init_name`, ...). Configs live in `script/stage_2_script/mlp_configs.py`: `BASELINE`, the one-change-at-a-time `ABLATIONS`, and `FINAL`.
- The ablation script resumes where it stopped if interrupted; `--fresh` starts over, and group names (e.g. `depth optimizer`) run only those groups.
- It runs on CPU by default (faster than MPS for a model this small); pass `device='mps'` or `'cuda'` to use a GPU.
- Outputs go to `result/stage_2_result/` (`ablation/` holds the per-run log, ablation table and per-group loss curves).
