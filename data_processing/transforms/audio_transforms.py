import os
import subprocess
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from scipy.io.wavfile import read
import librosa 
from scipy.io.wavfile import write


class AudioTransforms:

    def __init__(self):

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])

    def extract_audio(self, input_file, output_dir):
        output_file = os.path.join(output_dir, "audio.wav")
        subprocess.call(['ffmpeg', '-hwaccel', 'cuda', '-i', input_file,
                         '-codec:a', 'pcm_s16le', '-ac', '1', '-to', '1',
                         output_file, '-loglevel', 'quiet'])

        audio, sample_rate = librosa.load(output_file)
        audio = self.augment(audio, sample_rate=sample_rate)
        return audio

