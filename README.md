# speech_separation

This repository contains all the code we developed in the *ST7 - Séparation des Sources*.

* [Report](https://drive.google.com/file/d/1LE_m809DNKHvy86LSFxcnRmGVN52tXtP/view?usp=sharing)
* [Presentation](https://drive.google.com/file/d/1gt1x0pH34tnB_67X9mv1_4DvyoT65NEy/view?usp=sharing)
* [Dossier de laboratoire](https://docs.google.com/document/d/11V7hBRIvaO6X4SzRaiEeEM9nZDX8zH7QodhC265ZyU4/edit?usp=sharing)
* [Résultats de speech separation](https://drive.google.com/drive/folders/1p1Jg9Nxmfs9RRPvQZMv0aN4Gs3KYdPfg?usp=sharing)


## Jupyter notebooks
* `example_separation_ff.ipynb`: Source separation on female-female speaker pairs.
* `example_separation_mf.ipynb`: Source separation on male-female speaker pairs.
* `example_separation_ff.ipynb`: Source separation on male-male speaker pairs.
* `example_validation_mf.ipynb`: Model validation on male-female speaker pairs (calculation of metrics).

## Scripts
* `generate_dataset.py`: Generation of the mixed speaker dataset with utils.generate_dataset().
* `generate_json_pairs.py`: Generation of `pairs_train.json`, `pairs_val.json` and `pairs_test.json`.
* `model_validation.py`: Performs model validation.
* `train.sh`: Shell script to perform the training (consult the "Training" section on [SepFormer's documentation](https://huggingface.co/speechbrain/sepformer-wsj02mix)).
