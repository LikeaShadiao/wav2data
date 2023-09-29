# 该计算方法目前只适用于长度相等的两个信号，即音频样本数量相等的两个信号
# 但是将两个音频调整为长度相等是在这里实现了的
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample
import soundfile as sf


def calculate_snr(clean_file, noisy_file):
    # clean_rate, clean_data = wavfile.read(clean_file)
    # noisy_rate, noisy_data = wavfile.read(noisy_file)

    clean_data, clean_rate = sf.read(clean_file)
    noisy_data, noisy_rate = sf.read(noisy_file)

    print("clean noisy samples are: ", clean_rate, noisy_rate)


    print("clean noisy lengths are: ", len(clean_data), len(noisy_data))

    clean_data = clean_data[:len(noisy_data)]

    print("clean noisy lengths are: ", len(clean_data), len(noisy_data))


    noise = noisy_data - clean_data

    # 查看干净信号和噪声数据
    print("clean_data", clean_data)
    print("noise", noise)



    signal_power = abs(np.sum(clean_data ** 2))
    noise_power = abs(np.sum(noise ** 2))

    print("signal power : ", signal_power)
    print("noise_power : ", noise_power)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# 让代码被导入时不执行，只有在直接运行时才会执行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--clean_file", type = str, default = './enhanced.wav', help = 'something to input')
    parser.add_argument("--noise_file", type = str, default = './test.wav', help = 'something for noise')

    args = parser.parse_args()

    clean_file = args.clean_file
    noisy_file = args.noise_file
    print("clean : ", clean_file)
    print("noise : ", noisy_file)
    snr = calculate_snr(clean_file, noisy_file)
    print ("The signal-to-noise ratio is {:.2f} dB".format(snr))

