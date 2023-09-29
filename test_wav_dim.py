import soundfile as sf
from scipy.io import wavfile

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--inputfile", type = str, default = './noise_6.wav', help = 'something to input')

args = parser.parse_args()

# data = sf.read('./enhancedly_merge.wav')[0]
data = sf.read(args.inputfile)[0]

print('the dimension is: ', data.shape)

sample_rate, audio_data = wavfile.read(args.inputfile)

print('audio_data dimension is: ', audio_data.shape)
print('sample_rate is: ', sample_rate)
print('audio_data is: ', audio_data)

