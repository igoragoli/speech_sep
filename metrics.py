import torch
import mir_eval
import numpy as np

def si_snr(y_pred_batch, y_true_batch, reduction="mean"):
    """Computes the SI-SNR score.
    Arguments
    ---------
    y_pred_batch : The degraded signals.
    y_true_batch : The true signals.
    lens : The relative lengths of the waveforms within the batch.
    reduction : The type of reduction ("mean" or "batch") to use.
    """
    
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]
    SI_SNR = torch.zeros(batch_size)
    
    eps = 1e-6

    for i in range(0, batch_size):  # Run over mini-batches
        s_target = y_true_batch[i, 0 : y_pred_batch.shape[1]]
        s_estimate = y_pred_batch[i, 0 : y_pred_batch.shape[1]]

        # s_target = <ŝ, s>s / ||s||^2
        dot = torch.sum(s_estimate*s_target, dim=0, keepdim=True)
        s_target_energy = (
            torch.sum(s_target**2, dim=0, keepdim=True) + eps
        )
        proj = dot * s_target / s_target_energy

        # e_noise = ŝ - s_target
        e_noise = s_estimate - proj

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        si_snr_beforelog = torch.sum(proj**2, dim=0) / (
            torch.sum(e_noise**2, dim=0) + eps
        )
        SI_SNR[i] = 10 * torch.log10(si_snr_beforelog + eps)

    if reduction == "mean":
        return SI_SNR.mean()

    return SI_SNR

def sdr(y_pred_batch, y_true_batch, reduction="mean"):
    """Computes the SDR score.
    Arguments
    ---------
    y_pred_batch : The degraded signals.
    y_true_batch : The true signals.
    lens : The relative lengths of the waveforms within the batch.
    reduction : The type of reduction ("mean" or "batch") to use.
    """
    
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    y_pred_batch = np.array(y_pred_batch)
    y_true_batch = np.array(y_true_batch)
    
    (SDR, _, _, _)  = mir_eval.separation.bss_eval_sources(y_true_batch, y_pred_batch)

    if reduction == "mean":
        return SDR.mean()

    return SDR