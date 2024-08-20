#!/bin/bash

# continue=False # Continue or delete the parameters
# experiment_dir="./experiments/000_holdout_sanity_original"
# experiment_dir="./experiments/000_holdout_sanity_vqvae"
# experiment_dir="./experiments/001_holdout_original"
# experiment_dir="./experiments/002_holdout_vqvae"
# experiment_dir="./experiments/003_holdout_stylegan"
experiment_dir="./experiments/002_nested_cv_original"
# train_csv="original_train_subset.csv"
train_csv="original_train_data.csv"
# train_csv="vqvae_train_data.csv"
# train_csv="vqvae_train_data_subset.csv"
# train_csv="stylegan_train_data.csv"

# Check if the directory exists, if not create it
if [ ! -d "$experiment_dir" ]; then
  mkdir -p "$experiment_dir"
fi

# Copy the necessary files into the holdout directory
cp ./files_for_gandlf/gandlf_config.yaml "$experiment_dir/"
cp ./files_for_gandlf/$train_csv "$experiment_dir/"

# Run the GANDLF command
gandlf run \
  -c "$experiment_dir/gandlf_config.yaml" \
  -i "$experiment_dir/$train_csv" \
  -m "$experiment_dir/model_dir/" \
  --train \
  -d cuda


# hold out experiment with the same val and train test for sanity checks
# gandlf run \
#   -c "$experiment_dir/gandlf_config.yaml" \
#   -i "$experiment_dir/$train_csv,$experiment_dir/$train_csv" \
#   -m "$experiment_dir/model_dir/" \
#   --train \
#   -d cuda