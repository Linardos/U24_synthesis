#!/bin/bash

# continue=False # Continue or delete the parameters
# experiment_dir="./experiments/000_holdout_sanity_original"
# experiment_dir="./experiments/001_holdout_sanity_original_512x512"
experiment_dir="./experiments/001_holdout_original_256x256"
experiment_dir="./experiments/004_holdout_original_256x256_vgg"
experiment_dir="./experiments/004_cv_original_256x256_vgg"
# experiment_dir="./experiments/005_holdout_vqvae_256x256_vgg"
# experiment_dir="./experiments/005_cv_vqvae_256x256_vgg"
# experiment_dir="./experiments/005_holdout_vqvae_256x256_vgg"
experiment_dir="./experiments/006_holdout_original_256x256_flexinet"
# experiment_dir="./experiments/006_cv_original_256x256_flexinet"
# experiment_dir="./experiments/001_cv_original_256x256"
# experiment_dir="./experiments/002_cv_vqvae_256x256"
# experiment_dir="./experiments/002_holdout_vqvae"
# experiment_dir="./experiments/000_holdout_sanity_vqvae"
# experiment_dir="./experiments/001_holdout_original"
# experiment_dir="./experiments/002_holdout_vqvae"
# experiment_dir="./experiments/003_holdout_stylegan"

if [[ "$experiment_dir" == *"vqvae"* ]]; then
  if [[ "$experiment_dir" == *"holdout"* ]]; then
      train_csv="vqvae_0.9_train_split.csv"
      val_csv="vqvae_0.1_val_split_holdout.csv"
      cp ./files_for_gandlf/$val_csv "$experiment_dir/"
      echo training set for holdout with csv files $train_csv and $val_csv
  else
      train_csv="vqvae_train_data.csv"
      echo training set for cv: $train_csv
  fi
elif [[ "$experiment_dir" == *"original"* ]]; then
  if [[ "$experiment_dir" == *"holdout"* ]]; then
      train_csv="0.9_train_split.csv"
      val_csv="0.1_val_split_holdout.csv"
      cp ./files_for_gandlf/$val_csv "$experiment_dir/"
      echo training set for holdout with csv files $train_csv and $val_csv
  else
      train_csv="original_256_train_data.csv"
      echo training set for cv: $train_csv
  fi
elif [[ "$experiment_dir" == *"stylegan"* ]]; then
  if [[ "$experiment_dir" == *"holdout"* ]]; then
      train_csv="stylegan_0.9_train_split.csv"
      val_csv="stylegan_0.1_val_split_holdout.csv"
      cp ./files_for_gandlf/$val_csv "$experiment_dir/"
      echo training set for holdout with csv files $train_csv and $val_csv
  else
      train_csv="stylegan_train_data.csv"
      echo training set for cv: $train_csv
  fi
fi

export CUDA_VISIBLE_DEVICES=0

# Check if the directory exists, if not create it
if [ ! -d "$experiment_dir" ]; then
  mkdir -p "$experiment_dir"
fi

# Copy the necessary files into the holdout directory
cp ./files_for_gandlf/gandlf_config.yaml "$experiment_dir/"
cp ./files_for_gandlf/$train_csv "$experiment_dir/"
# 
echo starting training for $experiment_dir

# hold out experiment
if [[ "$experiment_dir" == *"holdout"* ]]; then
  echo running a holdout experiment...
  gandlf run \
    -c "$experiment_dir/gandlf_config.yaml" \
    -i "$experiment_dir/$train_csv,$experiment_dir/$val_csv" \
    -m "$experiment_dir/model_dir/" \
    --train \
    -d cuda
else
  echo running a cross-validation experiment...
  gandlf run \
    -c "$experiment_dir/gandlf_config.yaml" \
    -i "$experiment_dir/$train_csv" \
    -m "$experiment_dir/model_dir/" \
    --train \
    -d cuda
fi

# hold out experiment with the same val and train test for sanity checks
# gandlf run \
#   -c "$experiment_dir/gandlf_config.yaml" \
#   -i "$experiment_dir/$train_csv,$experiment_dir/$train_csv" \
#   -m "$experiment_dir/model_dir/" \
#   --train \
#   -d cuda