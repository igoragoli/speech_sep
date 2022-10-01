import json
from utils import select_dataset_files, form_pairs

# Selects files to considered at the training, validation and test sets
train_files, val_test_files = select_dataset_files(files_per_folder_train=5, files_per_folder_test=10, seconds_considered=5, split_train=0.8)

# Forms training pairs
pairs_train = form_pairs(train_files)

# Forms validation pairs
pairs_val_test = form_pairs(val_test_files)

# Creates json dictionaries
dict_train = {"length" : len(pairs_train), "pairs" : pairs_train}
dict_val_test = {"length" : len(pairs_val_test), "pairs" : pairs_val_test}

# Splits validation from tests
val_split = 0.2
split_idx = int(len(pairs_val_test)*val_split)
pairs_val = pairs_val_test[:split_idx]
pairs_test = pairs_val_test[split_idx:]

# Creates json dictionaries
dict_val = {"length" : len(pairs_val), "pairs" : pairs_val}
dict_test = {"length" : len(pairs_test), "pairs" : pairs_test}

# Creates json files
with open("pairs_test.json", "w") as fp:
    json.dump(dict_test, fp)

with open("pairs_val.json", "w") as fp:
    json.dump(dict_val, fp)

with open("pairs_train.json", "w") as fp:
    json.dump(dict_train, fp)
