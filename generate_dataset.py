import utils

utils.generate_dataset(
    files_per_folder=2,
    n_batches=8,
    n_pairs=1328,
    seconds_considered=2,
    fs=8000,
    split_percentages=[0, 0], 
    datapath='data/BREF80', 
    savepath='data/BREF-80-2mix',
    formed_pairs_path='data/JSON/pairs_train.json',
    formed_pairs_type='train'
)