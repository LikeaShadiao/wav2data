# 实现了将双通道音频转化为单通道（左）+ （右）
# 该函数一般只调用一次，修改这里的'input_audio_path'和convert函数调用时第二个参数即可

import librosa
import soundfile as sf
import numpy as np

def convert_to_mono(input_path, output_path, mode='average'):
    # Load the audio file
    audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
    print("audio_data is: ", audio_data)

    # 归一化
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert to mono based on the specified mode
    if mode == 'average':
        mono_audio = (audio_data[0] + audio_data[1]) / 2.0
    elif mode == 'left':
        mono_audio = audio_data[0]
        print("left is : ", mono_audio)
    elif mode == 'right':
        mono_audio = audio_data[1]
        print("right is : ", mono_audio)
    else:
        raise ValueError("Invalid mode. Use 'average', 'left', or 'right'.")

    # Save the mono audio to the output file
    sf.write(output_path, mono_audio, sample_rate)


# Specify the paths for input and output audio files
# input_audio_path = './noise_6.wav'
input_audio_path = './clean_6.wav'
output_audio_path = './testly_left.wav'



# Choose the mode: 'average', 'left', or 'right'
conversion_mode = 'left'

# Convert the audio to mono and save the output
convert_to_mono(input_audio_path, './clean_left.wav', mode='left')

convert_to_mono(input_audio_path, './clean_right.wav', mode='right')

# 输出两个音频数据查看
print ("left audio and right audio are : ")
laudio_data, lsample_rate = librosa.load('testly_left.wav', sr=None, mono=False)
raudio_data, rsample_rate = librosa.load('testly_right.wav', sr=None, mono=False)

print("left is : ", laudio_data)
print("right is : ", raudio_data)




