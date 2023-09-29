import soundfile as sf

def get_audio_dimensions(file_path):
    try:
        # 使用sf.read读取音频文件
        audio_data, sample_rate = sf.read(file_path)
        # 获取音频数据的维度
        dimensions = audio_data.shape if len(audio_data.shape) > 1 else (audio_data.shape[0], )
        return dimensions
    except Exception as e:
        print("Error reading audio file:", str(e))
        return None


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--inputfile", type = str, default = './noise_6.wav', help = 'something to input')

args = parser.parse_args()



# 指定音频文件路径
audio_file_path = args.inputfile

# 获取音频维度
dimensions = get_audio_dimensions(audio_file_path)

if dimensions:
    print('Audio dimensions:', dimensions)
else:
    print('Failed to retrieve audio dimensions.')
