import os
import random
import librosa

import numpy as np
import pickle as pkl

from src.datamodules.components._preprocess_wav import FeatureExtractor
from src.utils.rotation_conversions import *


dance_genre_dict = {
    "gBR": 0,
    "gPO": 1,
    "gLO": 2,
    "gMH": 3,
    "gLH": 4,
    "gHO": 5,
    "gWA": 6,
    "gKR": 7,
    "gJS": 8,
    "gJB": 9
}

extractor = FeatureExtractor()


def get_feature_sample(total_length, sample_length):
    return random.randint(0, total_length - sample_length)
    #
    # sample_idx = [pivot + i for i in range(sample_length)]
    # return sample_idx


def processing_music_list(music_data_path):
    music_dict = {}
    music_name = set([file.split('.')[0] for file in os.listdir(music_data_path)])

    for name in music_name:
        music_path = os.path.join(music_data_path, f'{name}.npy')
        if os.path.exists(music_path):
            music = np.load(music_path)
            music_dict.update({name: torch.from_numpy(music).float()})
        else:
            music_path = os.path.join(music_data_path, f'{name}.wav')
            music = wav_processing(music_path, name)
            music_dict.update({name: music})
    return music_dict


def processing_dance_list(motion_data_path):
    dance_dict = {}
    dance_list = []
    for pkl_file in os.listdir(motion_data_path):
        pkl_path = os.path.join(motion_data_path, pkl_file)
        dance = pkl_processing(pkl_path)
        dance_dict.update({pkl_file.split('.')[0]: dance})
        dance_list.append(pkl_file.split('.')[0])
    return dance_dict, dance_list


def wav_processing(wav_path, audio_name):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else:
            assert False, audio_name

    audio, _ = librosa.load(wav_path, sr=SR)
    melspe_db = extractor.get_melspectrogram(audio, SR)

    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave=7 if SR == 15360 * 2 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, SR)
    tempogram = extractor.get_tempogram(onset_env, SR)
    onset_beat = extractor.get_onset_beat(onset_env, SR)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc,  # 20
        mfcc_delta,  # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt,  # 12
        onset_env,  # 1
        onset_beat,  # 1
        tempogram
    ], axis=0)

    # mfcc, #20
    # mfcc_delta, #20

    # chroma_cqt, #12
    # onset_env, # 1
    # onset_beat, #1

    feature = feature.transpose(1, 0)

    save_path = wav_path.split('.')[0]
    np.save(f'{save_path}.npy', feature)
    audio_feature = torch.from_numpy(feature).float()
    return audio_feature


def pkl_processing(pkl_path):
    motion = pkl.load(open(pkl_path, "rb"))

    smpl_poses = torch.from_numpy(motion['smpl_poses']).float()
    smpl_trans = torch.from_numpy(motion['smpl_trans'] / motion['smpl_scaling']).float()

    # ret = torch.cat([smpl_poses, smpl_trans], dim=1)
    # ret = smpl_poses
    return [smpl_poses, smpl_trans]


