# plant3_clean

Trained on the 67-plant train split of `script_point_transformer/soybean_split.json`
only, so unlike `../plant3` (trained on all 78) it has never seen the 11 test plants.

- Recipe `fixed` (`config_fixed.py`); use `--recipe fixed` at inference.
- Realigned dataset (`stem_axis: x`), 1 GPU, batch 4, 100 passes over the data.
- Held-out, single view: pooled mIoU **0.470**, point accuracy **0.963**.

```bash
python script_point_transformer/run.py test --recipe fixed --tta single \
  --weight third_party/PointTransformer_V3/Pointcept/exp/soybean3d/plant3_clean/model/model_last.pth \
  --output outputs/point_transformer/plant3_clean_test
```
