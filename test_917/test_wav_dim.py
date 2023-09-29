import soundfile as sf
from scipy.io import wavfile
import librosa

# 查看含噪/干净信号的两个通道
print ("left audio and right audio are : ")
laudio_data, lsample_rate = librosa.load('clean_left.wav', sr=None, mono=False)
raudio_data, rsample_rate = librosa.load('clean_right.wav', sr=None, mono=False)

print("left is : ", laudio_data)
print("right is : ", raudio_data)






import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--inputfile", type = str, default = './noise_6.wav', help = 'something to input')

args = parser.parse_args()

# data = sf.read('./enhancedly_merge.wav')[0]
# data = librosa.load(args.inputfile)[0]
# audio_data, sample_rate = librosa.load(args.inputfile, sr=None, mono=False)
audio_data, sample_rate = sf.read(args.inputfile)

print('the dimension is: ', audio_data.shape)

# sample_rate, audio_data = wavfile.read(args.inputfile)

print('audio_data dimension is: ', audio_data.shape)
print('sample_rate is: ', sample_rate)
print('audio_data is: ', audio_data)

