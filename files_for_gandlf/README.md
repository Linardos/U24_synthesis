construct_labelCSV.py is used to create the csv files gandlf needs.

make a subset of the csv with this:
(head -n 10 vqvae_train_data.csv && tail -n 10 vqvae_train_data.csv) > vqvae_train_data_subset.csv
(head -n 10 stylegan_train_data.csv && tail -n 10 stylegan_train_data.csv) > stylegan_train_data_subset.csv