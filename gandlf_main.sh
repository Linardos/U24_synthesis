gandlf run \
  -c ./experiments/001_holdout/gandlf_config.yaml \
  -i "/home/locolinux/U24_Synthesis/experiments/001_holdout/train_subset.csv,/home/locolinux/U24_Synthesis/experiments/001_holdout/train_subset.csv" \
  -m ./experiments/001_holdout/model_dir/ \
  --infer \
  -d cuda