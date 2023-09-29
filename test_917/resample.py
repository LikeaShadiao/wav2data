# 将采样为8khz的干净信号重采样为16khz
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--inputfile", type = str, default = './noise_6.wav', help = 'something to input')
parser.add_argument("--outputfile", type = str, default = './noise_16k.wav', help = 'something to output')

args = parser.parse_args()


# Load the 8000 Hz sample rate audio
input_sample_rate, audio_data = wavfile.read(args.inputfile)

# Define the target sample rate
# target_sample_rate = 16000
target_sample_rate = 8000

num_resample = int(len(audio_data) * (target_sample_rate / input_sample_rate))

# Resample the audio data
# resampled_audio_data = resample(audio_data, 195200)
resampled_audio_data = resample(audio_data, num_resample)

# Save the resampled audio as a new WAV file
output_file = args.outputfile
wavfile.write(output_file, target_sample_rate, resampled_audio_data.astype(np.int16))

print('Resampled audio saved to', output_file)
