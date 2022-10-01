import os
import random
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
import json

from metrics import si_snr, sdr

def select_dataset_files(files_per_folder_train=2, files_per_folder_test=5, seconds_considered=5, split_train=0.8, datapath='data/BREF80'):
    """Selects speakers for the training and test sets.
    Arguments
    ---------
    files_per_folder_training : number of audio extracts from each speaker to be considered in training.
    files_per_folder_test : number of audio extracts from each speaker to be considered in test.
    seconds_considered : length (in seconds) of the audios.
    split_train : ratio of speakers to be considered in training.
    datapath : folder containing BREF-80 dataset.
    """

    # Creates list of valid folders
    folders = os.listdir(datapath)
    n_train_folders = int(len(folders)*split_train)

    random.shuffle(folders)
    train_folders = folders[:n_train_folders]
    test_val_folders = folders[n_train_folders:]

    # Selects files in the data/BREF80 folder
    train_files = []
    for folder in train_folders:
        files = os.listdir(os.path.join(datapath, folder))
        if len(files) >= files_per_folder_train:  # checks if there are enough audio files in the folder
            # gets random files from the folder
            random_files = random.sample(files, files_per_folder_train)
            for file in random_files:
                folder_file = os.path.join(folder, file)
                fs, audio = wavfile.read(
                    os.path.join(datapath, folder_file)
                )
                if len(audio)/fs > seconds_considered:  # checks if audio has enough length
                    train_files.append(folder_file)

    # Selects files in the data/BREF80 folder
    test_val_files = []
    for folder in test_val_folders:
        files = os.listdir(os.path.join(datapath, folder))
        if len(files) >= files_per_folder_test:  # checks if there are enough audio files in the folder
            # gets random files from the folder
            random_files = random.sample(files, files_per_folder_test)
            for file in random_files:
                folder_file = os.path.join(folder, file)
                fs, audio = wavfile.read(
                    os.path.join(datapath, folder_file)
                )
                if len(audio)/fs > seconds_considered:  # checks if audio has enough length
                    test_val_files.append(folder_file)

    return train_files, test_val_files

def select_files(files_per_folder=2, seconds_considered=5, specify_sex=None, datapath='data/BREF80'):
    """Selects speakers without considering the formation of training/validation/test sets.
    Arguments
    ---------
    files_per_folder : number of audio extracts from each speaker to be considered.
    seconds_considered : length (in seconds) of the audios.
    specify_sex : 'M', 'F' or None.
    datapath : folder containing BREF-80 dataset.
    """

    # creates list of valid folders
    folders = os.listdir(datapath)
    if specify_sex is not None:  # specify_sex: 'M', 'F' or None
        folders = [folder for folder in folders if folder[2] == specify_sex]

    # Selects files in the data/BREF80 folder
    selected_files = []
    for folder in folders:
        files = os.listdir(os.path.join(datapath, folder))
        if len(files) >= files_per_folder:  # checks if there are enough audio files in the folder
            # gets random files from the folder
            random_files = random.sample(files, files_per_folder)
            for file in random_files:
                folder_file = os.path.join(folder, file)
                fs, audio = wavfile.read(
                    os.path.join(datapath, folder_file)
                )
                if len(audio)/fs > seconds_considered:  # checks if audio has enough length
                    selected_files.append(folder_file)

    return selected_files

def form_pairs(selected_files):
    """Forms all pairs possible (excluding pairs including the same speaker) from a given list of selected files.
    Arguments
    ---------
    selected_files : list of files to be considered.
    """

    formed_pairs = []
    remaining_files = selected_files.copy()
    for file in selected_files:
        remaining_files.pop()
        for remaining_file in remaining_files:
            if file[:3] != remaining_file[:3]: # Checks if the speakers are not the same
                formed_pairs.append(file + '-' + remaining_file)

    return formed_pairs

def separate_pair(pair, model, device, seconds_considered=2, datapath='data/BREF80'):

    """Computes estimated sources given a pair of original sources.
    Arguments
    ---------
    pair : pair of original sources to be separated
    model : SepFormer separator model
    device : device in which the model is run (CPU or GPU)
    seconds_considered : length (in seconds) of the original sources to be considered
    datapath : folder containing BREF-80 dataset
    """

    # load audios as arrays
    file1, file2 = pair.split('-')
    audio1, fs1 = torchaudio.load(os.path.join(datapath, file1))
    audio2, fs2 = torchaudio.load(os.path.join(datapath, file2))

    assert fs1 == fs2

    # Clip audios at desired length
    audio1 = audio1[0, :seconds_considered*fs1]
    audio2 = audio2[0, :seconds_considered*fs2]

    # Adding audios and normalizing them
    mix = audio1 + audio2
    mix = 2*(mix - torch.min(mix))/(torch.max(mix) - torch.min(mix)) - 1
    mix = torch.unsqueeze(mix, dim=0)
    mix = mix.to(device)

    # Separating audios (using SpeechBrain`s model)
    es = separate_audios(mix, fs1, model)
    es1 = es[0, :, 0]
    es2 = es[0, :, 1]

    # Converting tensors to numpy arrays
    mix_np = mix.cpu().numpy()
    audio1_np = audio1.cpu().numpy()
    audio2_np = audio2.cpu().numpy()
    es1_np = es1.cpu().numpy()
    es2_np = es2.cpu().numpy()

    # Original sources
    sources = [audio1_np, audio2_np]

    # Estimated sources
    predictions = [es1_np, es2_np]
    
    return mix_np[0], sources, predictions

def evaluate_separation(sources, predictions):

    """Calculates quality metrics (SI-SNR and SDR) given separation.
    Arguments
    ---------
    sources : original audio sources.
    predictions : estimated audio sources predicted by separator.
    """

    # Matching original sources to estimated sources
    set1, set2 = compare_sources_with_predictions(sources, predictions)

    set11 = torch.from_numpy(set1[0, :])
    set12 = torch.from_numpy(set1[1, :])
    set21 = torch.from_numpy(set2[0, :])
    set22 = torch.from_numpy(set2[1, :])

    set11 = torch.unsqueeze(set11, dim=0)
    set12 = torch.unsqueeze(set12, dim=0)
    set21 = torch.unsqueeze(set21, dim=0)
    set22 = torch.unsqueeze(set22, dim=0)

    # Calculating SI-SNR
    si_snr1 = si_snr(set11, set12).cpu().numpy()
    si_snr2 = si_snr(set21, set22).cpu().numpy()

    # Calculating SDR
    sdr1 = sdr(set11, set12)
    sdr2 = sdr(set21, set22)

    return si_snr1, si_snr2, sdr1, sdr2

def create_gender_mask(pairs, gender_mix):

    """Creates mask used to sort results by the sex of the speakers.
    Arguments
    ---------
    pairs : pairs to be considered.
    gender_mix : sex combination to be considered ("MM", "FF" and "MF"/"FM").
    """

    mask = []
    for pair in pairs:

        file1, file2 = pair.split('-')
        gender1 = file1[2]
        gender2 = file2[2]

        if gender1=='M' and gender2=='M':
            if gender_mix=="MM":
                mask.extend((True, True))
            else:
                mask.extend((False, False))

        elif gender1=='F' and gender2=='F':
            if gender_mix=="FF":
                mask.extend((True, True))
            else:
                mask.extend((False, False))

        else:
            if gender_mix=="MF" or gender_mix=="FM":
                mask.extend((True, True))
            else:
                mask.extend((False, False))

    return mask

def generate_dataset(files_per_folder=2, n_pairs=None, n_batches=1, 
    seconds_considered=1, fs=8000,
    split_percentages=[0.8, 0.1], datapath='data/BREF80', savepath='BREF-80-2mix', 
    formed_pairs_path='pairs_train.json', formed_pairs_type='train'):
    """Generates a dataset with a structure compatible with SpeechBrain's training script 
    on SepFormer. The correct folder structure is described in our report.
    Arguments
    ---------
    files_per_folder : number of audio extracts from each speaker to be considered.
    n_pairs : number of audio pairs that should be in the dataset.
    n_batches : in how many batches the saving of audio files will be done.
    seconds_considered : length (in seconds) of the audios.
    fs : Sampling frequency of the audio files.
    split_percentages: train-val-test split percentages. Ex.: [0.8, 0.1] means that 80% of the data
        will form the training set, 10% will form the validation set, and the remaining 10% will form
        the test set.
    datapath : Path where the BREF80 speech data should be fetched.
    savepath : Path where the generated dataset should be saved.
    formed_pairs_path : Path to a JSON file containing the list of formed pairs to be saved.
    formed_pairs_type : Type of set related to the JSON formed pair list ('train', 'val' or 'test').
        If no argument is specified, the function will split the formed pair list onto a training,
        validation and test sets according to the 'split_percentages' argument.
    """

    if formed_pairs_path:
        with open(formed_pairs_path) as fp:
            formed_pairs_dict = json.load(fp)
            formed_pairs = formed_pairs_dict["pairs"]
        if formed_pairs_type == 'train':
            split_percentages = [1, 0]
        elif formed_pairs_type == 'val':
            split_percentages = [0, 1]
        elif formed_pairs_type == 'test':
            split_percentages == [0, 0]
        elif formed_pairs_type:
            raise ValueError("The argument 'formed_pairs_type' is not 'train', 'val' neither 'test'")

    else:
        # Selects files in the data/BREF80 folder
        selected_files = select_files(files_per_folder, seconds_considered, 
            specify_sex=None, datapath=datapath)

        # Forms pairs from the selected files 
        formed_pairs = form_pairs(selected_files)

    # Clip pairs if necessary
    if n_pairs:
        if n_pairs > len(formed_pairs):
            n_pairs = len(formed_pairs)
        formed_pairs = formed_pairs[:n_pairs]
    else:
        n_pairs = len(formed_pairs)
                
    # Mixes audio pairs. The audio pairs are stored on 'audios', while the original sources are
    # stored on 'source_audios'
    n_pairs_per_batch = n_pairs//n_batches

    print("Mixing/Saving audio pairs")
    for l in range(0, n_batches):
        print("Mixing batch {}/{}".format(l + 1, n_batches))
        init_audios = True
        for i, pair in enumerate(formed_pairs[l*n_pairs_per_batch:(l + 1)*n_pairs_per_batch], start=1):
            print("{}/{}".format(i, n_pairs_per_batch), end="\r")
            file1, file2 = pair.split('-')

            source1, fs1 = torchaudio.load(
                os.path.join(datapath, file1)
            )
            source2, fs2 = torchaudio.load(
                os.path.join(datapath, file2)
            )
            source = torch.stack(
                [source1[0, :seconds_considered*fs1], 
                source2[0, :seconds_considered*fs2]], 
                dim=0
            )

            mixture = source1[0, :seconds_considered*fs1] + source2[0, :seconds_considered*fs2]
            mixture = 2*(mixture - torch.min(mixture))/(torch.max(mixture) - torch.min(mixture)) - 1
            mixture = torch.unsqueeze(mixture, dim=0)

            if init_audios:
                sources_batch = source
                mixtures_batch = mixture
                init_audios = False
            else:
                sources_batch = torch.cat([sources_batch, source], dim=0)
                mixtures_batch = torch.cat([mixtures_batch, mixture], dim=0)

        print("Saving batch {}/{}".format(l + 1, n_batches))
        save_mixtures(sources_batch, mixtures_batch, l*n_pairs_per_batch, split_percentages, fs, savepath)

def save_mixtures(sources_batch, mixtures_batch, file_number_start=0,
    split_percentages=[0.8, 0.1], fs=8000, savepath='BREF-80-2mix'):
    """Given a batch of sources and a batch of mixtures, save audio files in the correct
    folders. The correct folder structure is described in our report.
    Arguments
    ---------
    sources_batch : batch of source audios (2-dimensional np.array or torch.Tensor, where different
        audios are stacked along the lines of the matrix. The odd lines correspond to the sources 
        labeled as '1', where the even lines correspond to the sources labeled as '2').
    mixtures_batch : batch of mixture audios (2-dimensional np.array or torch.Tensor, where different
        audios are stacked along the lines of the matrix).
    file_number_start : at which number should the file naming start.
    split_percentages : train-val-test split percentages. Ex.: [0.8, 0.1] means that 80% of the data
        will form the training set, 10% will form the validation set, and the remaining 10% will form
        the test set.
    fs : Sampling frequency of the audio files.
    savepath : Path where the generated dataset should be saved.
    """
    
    set_types=["train", "valid", "test"]
    folder_names=["source1", "source2", "mixture"]

    # Create directories
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    for set_type in set_types:
        if not os.path.exists(os.path.join(savepath, set_type)):
                os.mkdir(os.path.join(savepath, set_type))

        for folder_name in folder_names:
            if not os.path.exists(os.path.join(savepath, set_type, folder_name)):
                os.mkdir(os.path.join(savepath, set_type, folder_name) )

    n_mix = mixtures_batch.shape[0]

    # Save files
    for i in range(0, n_mix):
        if i < n_mix*split_percentages[0]:
            # Training set
            set_type = "train"
            for folder_name in folder_names:
                folder_path = os.path.join(savepath, set_type, folder_name) 
                
                if folder_name == "source1":
                    file = sources_batch[2*i, :]
                elif folder_name == "source2":
                    file = sources_batch[2*i + 1, :]
                elif folder_name == "mixture":
                    file = mixtures_batch[i, :]
                
                # Normalize audio between -1 and 1 -- Already done
                #file = 2*(file - torch.min(file))/(torch.max(file) - torch.min(file)) - 1

                file_name = format(i + file_number_start, '05d') + ".wav"
                torchaudio.save(
                    os.path.join(folder_path, file_name), 
                    torch.unsqueeze(file, dim=0), 
                    fs
                )

        elif i < n_mix*sum(split_percentages):
            # Validation set
            set_type = "valid"
            for folder_name in folder_names:
                folder_path = os.path.join(savepath, set_type, folder_name)

                if folder_name == "source1":
                    file = sources_batch[2*i, :]
                elif folder_name == "source2":
                    file = sources_batch[2*i + 1, :]
                elif folder_name == "mixture":
                    file = mixtures_batch[i, :]
                
                # Normalize audio between -1 and 1 -- Already done
                #file = 2*(file - torch.min(file))/(torch.max(file) - torch.min(file)) - 1

                file_name = format(i + file_number_start, '05d') + ".wav"
                torchaudio.save(
                    os.path.join(folder_path, file_name), 
                    torch.unsqueeze(file, dim=0), 
                    fs
                )   
        else:
            # Test
            set_type = "test"
            for folder_name in folder_names:
                folder_path = os.path.join(savepath, set_type, folder_name)

                if folder_name == "source1":
                    file = sources_batch[2*i, :]
                elif folder_name == "source2":
                    file = sources_batch[2*i + 1, :]
                elif folder_name == "mixture":
                    file = mixtures_batch[i, :]
                
                # Normalize audio between -1 and 1 -- Already done
                #file = 2*(file - torch.min(file))/(torch.max(file) - torch.min(file)) - 1

                file_name = format(i + file_number_start, '05d') + ".wav"
                torchaudio.save(
                    os.path.join(folder_path, file_name), 
                    torch.unsqueeze(file, dim=0), 
                    fs
                )

def resample_audios(audios, orig_freq, new_freq, device, verbose=False):

    """Resamples audios given as argument.
    Arguments
    ---------
    audios : audios to be resampled.
    orig_freq : original frequencies of the audios.
    new_freq : new/desired frequency of the audios.
    device : device in which the model is run (torch.device object representing CPU or GPU).
    verbose : print (True) or not (False) messages along the resampling.
    """

    if verbose:
        print("Resampling audio(s) from {} Hz to {} Hz".format(orig_freq, new_freq))
    tf = torchaudio.transforms.Resample(orig_freq, new_freq).to(device)

    init_audios = True
    for i in range(0, audios.shape[0]):
        if verbose:
            print("Resampling audio {} of {}".format(i + 1, audios.shape[0]))
        audio = torch.unsqueeze(audios[i, ...], dim=0)
        resampled_audio = tf(audio)

        if init_audios:
            resampled_audios = resampled_audio
            init_audios = False
        else:
            resampled_audios = torch.cat(
                [resampled_audios, resampled_audio], dim=0)

    return resampled_audios


def separate_audios(audios, fs_audio, model, verbose=False):

    """Separates audios given as arguments
    Arguments
    ---------
    audios : audios to be separated
    fs_audio : sampling frequency of the audio (must be the same for every audio)
    model : SepFormer separator model
    verbose :print (True) or not (False) messages along the resampling
    """

    audios = audios.to(model.device)
    fs_model = model.hparams.sample_rate

    resampled_audios = resample_audios(
        audios, fs_audio, fs_model, model.device)

    if verbose:
        print("Separating sources")

    est_sources = model.separate_batch(resampled_audios)
    est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]
    return est_sources

def compare_sources_with_predictions(sources, predictions):

    """Matches estimated sources to their corresponding original sources.
    Arguments
    ---------
    sources : original audio sources.
    predictions : estimated audio sources predicted by separator.
    """

    # Downsampling original sources
    source1 = sources[0][::2]
    source2 = sources[1][::2]
    prediction1 = predictions[0]
    prediction2 = predictions[1]

    # Calculating correlation between first original source and both estimated sources
    xcorr11 = np.sum(np.abs(np.correlate(source1, prediction1, 'full')))
    xcorr12 = np.sum(np.abs(np.correlate(source1, prediction2, 'full')))
    xcorr1 = [xcorr11, xcorr12]
    ind1 = np.argmax(xcorr1)

    # Calculating correlation between second original source and both estimated sources
    xcorr21 = np.sum(np.abs(np.correlate(source2, prediction1, 'full')))
    xcorr22 = np.sum(np.abs(np.correlate(source2, prediction2, 'full')))
    xcorr2 = [xcorr21, xcorr22]
    ind2 = np.argmax(xcorr2)

    # Matching estimated sources to original sources
    if ind1 != ind2:
        set1 = np.stack((source1, predictions[ind1]))
        set2 = np.stack((source2, predictions[ind2]))

    else:
        if np.abs(xcorr11-xcorr12) > np.abs(xcorr21-xcorr22):
            set1 = np.stack((source1, predictions[ind1]))
            set2 = np.stack((source2, predictions[not(ind1)]))
        else:
            set2 = np.stack((source2, predictions[ind2]))
            set1 = np.stack((source1, predictions[not(ind2)]))

    return set1, set2
