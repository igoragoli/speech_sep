import os
import random
import numpy as np
import torch
from tqdm import tqdm
import json

from speechbrain.pretrained import SepformerSeparation as separator
from utils import separate_pair, evaluate_separation, create_gender_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("pairs_test.json", "r") as fp:
    pairs_dict = json.load(fp)

pairs = pairs_dict["pairs"]
#pairs = random.sample(pairs, 50)

if device == torch.device("cuda"):
    model = separator.from_hparams(source="speechbrain/sepformer-whamr",
                                   savedir='pretrained_models/sepformer-whamr', run_opts={"device": "cuda"})
else:
    model = separator.from_hparams(
        source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

si_snr_list = []
sdr_list = []
for pair in tqdm(pairs, "Evaluating pairs"):
    _, sources, predictions = separate_pair(pair, model, device, seconds_considered=2)
    si_snr1, si_snr2, sdr1, sdr2  = evaluate_separation(sources, predictions)
    si_snr_list.extend([si_snr1, si_snr2])
    sdr_list.extend([sdr1, sdr2])

si_snr_np = np.stack(si_snr_list)
sdr_np = np.stack(sdr_list)

os.makedirs("before_training", exist_ok=True)
np.save("before_training/si_snr_before", si_snr_np)
np.save("before_training/sdr_before", sdr_np)

print("\n----- General Results -----")
print("Median - SI-SNR: " + str(np.median(si_snr_np)))
print("Mean - SI-SNR: " + str(np.mean(si_snr_np)))
print("Median - SDR: " + str(np.median(sdr_np)))
print("Mean - SDR: " + str(np.mean(sdr_np)))

print("\n----- Male-Male Results -----")
mask_mm = create_gender_mask(pairs, "MM")
si_snr_mm = si_snr_np[np.array(mask_mm)]
sdr_mm = sdr_np[np.array(mask_mm)]
np.save("before_training/si_snr_mm_before", si_snr_mm)
np.save("before_training/sdr_mm_before", sdr_mm)
print("Median - SI-SNR: " + str(np.median(si_snr_mm)))
print("Mean - SI-SNR: " + str(np.mean(si_snr_mm)))
print("Median - SDR: " + str(np.median(sdr_mm)))
print("Mean - SDR: " + str(np.mean(sdr_mm)))

print("\n----- Female-Female Results -----")
mask_ff = create_gender_mask(pairs, "FF")
si_snr_ff = si_snr_np[np.array(mask_ff)]
sdr_ff = sdr_np[np.array(mask_ff)]
np.save("before_training/si_snr_ff_before", si_snr_ff)
np.save("before_training/sdr_ff_before", sdr_ff)
print("Median - SI-SNR: " + str(np.median(si_snr_ff)))
print("Mean - SI-SNR: " + str(np.mean(si_snr_ff)))
print("Median - SDR: " + str(np.median(sdr_ff)))
print("Mean - SDR: " + str(np.mean(sdr_ff)))

print("\n----- Male-Female Results -----")
mask_mf = create_gender_mask(pairs, "MF")
si_snr_mf = si_snr_np[np.array(mask_mf)]
sdr_mf = sdr_np[np.array(mask_mf)]
np.save("before_training/si_snr_mf_before", si_snr_mf)
np.save("before_training/sdr_mf_before", sdr_mf)
print("Median - SI-SNR: " + str(np.median(si_snr_mf)))
print("Mean - SI-SNR: " + str(np.mean(si_snr_mf)))
print("Median - SDR: " + str(np.median(sdr_mf)))
print("Mean - SDR: " + str(np.mean(sdr_mf)))
