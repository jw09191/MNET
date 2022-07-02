import librosa
import numpy as np


class FeatureExtractor:
    @staticmethod
    def get_melspectrogram(audio, sample_rate):
        melspe = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        melspe_db = librosa.power_to_db(melspe, ref=np.max)
        return melspe_db

    @staticmethod
    def get_hpss(audio):
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
        return audio_harmonic, audio_percussive

    @staticmethod
    def get_mfcc(melspe_db):
        mfcc = librosa.feature.mfcc(S=melspe_db)
        return mfcc

    @staticmethod
    def get_mfcc_delta(mfcc):
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        return mfcc_delta

    @staticmethod
    def get_mfcc_delta2(mfcc):
        mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
        return mfcc_delta_delta

    @staticmethod
    def get_harmonic_melspe_db(audio_harmonic, sr):
        harmonic_melspe = librosa.feature.melspectrogram(audio_harmonic, sr=sr)
        harmonic_melspe_db = librosa.power_to_db(harmonic_melspe, ref=np.max)
        return harmonic_melspe_db

    @staticmethod
    def get_percussive_melspe_db(audio_percussive, sr):
        percussive_melspe = librosa.feature.melspectrogram(audio_percussive, sr=sr)
        percussive_melspe_db = librosa.power_to_db(percussive_melspe, ref=np.max)
        return percussive_melspe_db

    @staticmethod
    def get_chroma_cqt(audio_harmonic, sr, octave=7):
        chroma_cqt_harmonic = librosa.feature.chroma_cqt(y=audio_harmonic, sr=sr, n_octaves=octave)
        return chroma_cqt_harmonic

    @staticmethod
    def get_chroma_stft(audio_harmonic, sr):
        chroma_stft_harmonic = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr)
        return chroma_stft_harmonic

    @staticmethod
    def get_tonnetz(audio_harmonic, sr):
        tonnetz = librosa.feature.tonnetz(y=audio_harmonic, sr=sr)
        return tonnetz

    @staticmethod
    def get_onset_strength(audio_percussive, sr):
        onset_env = librosa.onset.onset_strength(audio_percussive, aggregate=np.median, sr=sr)
        return onset_env

    @staticmethod
    def get_tempogram(onset_env, sr):
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        return tempogram

    @staticmethod
    def get_onset_beat(onset_env, sr):
        onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        beats_one_hot = np.zeros(len(onset_env))
        peaks_one_hot = np.zeros(len(onset_env))
        for idx in onset_beats:
            beats_one_hot[idx] = 1
        for idx in peaks:
            peaks_one_hot[idx] = 1

        beats_one_hot = beats_one_hot.reshape(1, -1)
        peaks_one_hot = peaks_one_hot.reshape(1, -1)

        return beats_one_hot, peaks_one_hot
