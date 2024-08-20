# python gandlf_constructCSV -i /mnt/c/Datasets/BCS-DBT-Szymon/original -c .nii.gz -o ./original_train_data.csv # output CSV to be used for training
gandlf construct-csv -i /mnt/c/Datasets/BCS-DBT-Szymon/vqvae -c .nii.gz -o ./vqvae_train_data.csv # output CSV to be used for training
# this doesnt work at all lmao just use construct_labelCSV.py

make a subset of the csv with this:
(head -n 10 vqvae_train_data.csv && tail -n 10 vqvae_train_data.csv) > vqvae_train_data_subset.csv
(head -n 10 stylegan_train_data.csv && tail -n 10 stylegan_train_data.csv) > stylegan_train_data_subset.csv
