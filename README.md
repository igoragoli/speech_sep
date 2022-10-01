# speech_sep: Speech Separation

This project is a second year engineering school (equivalent to senior year of college in the US system / 1st year of Master in the UK system) project of CentraleSupelec engineering school (Paris-Saclay University) in France.

A source separation problem was posed by Orange. In the given problem, assistant robots used by the company, capable of performing natural language processing and interacting with clients in French, should be able to follow a conversation with a speaker in an environment with many different speakers talking at the same time. From this main issue, considering the constraints in our project, a simpler and more direct goal was stated:

**"Given a mixed signal constructed from two different sources, find the best estimate of the original sources."**

The [SepFormer](https://huggingface.co/speechbrain/sepformer-wsj02mix) model was applied in a custom-made french speech corpus based on [BREF-80](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0006/).

## Project Documents
* [Report](https://drive.google.com/file/d/1LE_m809DNKHvy86LSFxcnRmGVN52tXtP/view?usp=sharing)
* [Presentation](https://drive.google.com/file/d/1gt1x0pH34tnB_67X9mv1_4DvyoT65NEy/view?usp=sharing)
* [Speech separation audio results](https://drive.google.com/drive/folders/1p1Jg9Nxmfs9RRPvQZMv0aN4Gs3KYdPfg?usp=sharing)

## Jupyter Notebooks
* `example_separation_ff.ipynb`: Source separation on female-female speaker pairs.
* `example_separation_mf.ipynb`: Source separation on male-female speaker pairs.
* `example_separation_ff.ipynb`: Source separation on male-male speaker pairs.
* `example_validation_mf.ipynb`: Model validation on male-female speaker pairs (calculation of metrics).

## Scripts
* `generate_dataset.py`: Generation of the mixed speaker dataset with utils.generate_dataset().
* `generate_json_pairs.py`: Generation of `pairs_train.json`, `pairs_val.json` and `pairs_test.json`.
* `model_validation.py`: Performs model validation.
* `train.sh`: Shell script to perform the training (consult the "Training" section on [SepFormer's documentation](https://huggingface.co/speechbrain/sepformer-wsj02mix)) on CentraleSupélec's GPU clusters.
