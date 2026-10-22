# ECS 170 Stage 2 — Aron data pipeline

From the project root, install `numpy` and run:

```bash
python -m pip install numpy
python -m script.stage_2_script.check_data
```

The loader returns `{'train': {'X': ..., 'y': ...}, 'test': {'X': ..., 'y': ...}}` with float32 pixel features scaled to [0,1] and int64 labels 0–9. No re-splitting or cross-validation.

**For Kartik:** Adapt `local_code/stage_2_code/Method_MLP.py` to accept 784 input features and produce 10 logits. With `CrossEntropyLoss`, do not apply Softmax to the logits before computing loss. For large training data, use mini-batches instead of full-batch training. Wire the script to `Setting_Predefined_Split` and the Stage 2 dataset path. `Setting_Predefined_Split` follows the template's save/evaluate API. The copied Stage 1 MLP is deliberately not rewritten here because that is Kartik's workstream.

**For teammate integration:** `Dataset_Loader.load()` returns both predefined splits. Use `Setting_Predefined_Split`, not `Setting_KFold_CV` or `Setting_Train_Test_Split`. Set result folder to `result/stage_2_result/`.
