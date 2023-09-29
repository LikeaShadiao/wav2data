# 将处理后的做通道音频与原音频的右通道合成新的音频

import numpy as np
import soundfile as sf

def combine_audio_as_stereo(left_mono_audio, right_stereo_audio, output_path):
    # Load the audio data
    left_audio, left_sample_rate = sf.read(left_mono_audio)
    right_audio, right_sample_rate = sf.read(right_stereo_audio)

    # Ensure both audio clips have the same length
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    print("length is: ", min_length)
    print("left audio : ", left_audio)
    print("right audio : ", right_audio)

    stereo_audio = np.column_stack((left_audio, right_audio))

    print("final is : ", stereo_audio)
    print("shape is : ", stereo_audio.shape)


    # Save the stereo audio to the output file
    sf.write(output_path, stereo_audio, right_sample_rate)  # Use left sample rate

# Specify the paths for input audio files (left mono and right stereo)
left_mono_audio_path = './enhancedly_left.wav'
right_stereo_audio_path = './testly_right.wav'

# Specify the output file path for the combined stereo audio
output_stereo_audio_path = './enhancedly_merge.wav'

# Combine audio as stereo (left channel is left_mono_audio, right channel is right_stereo_audio)
combine_audio_as_stereo(left_mono_audio_path, right_stereo_audio_path, output_stereo_audio_path)

