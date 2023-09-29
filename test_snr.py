import numpy as np
import scipy.io.wavfile as wavfile

def calculate_snr(clean_file, noisy_file):
    clean_rate, clean_data = wavfile.read(clean_file)
    noisy_rate, noisy_data = wavfile.read(noisy_file)

    if clean_rate != noisy_rate:
        raise ValueError("Sampling rate mismatch between clean and noisy files")

    if len(clean_data) != len(noisy_data):
        raise ValueError("Length mismatch between clean and noisy files")

    noise = noisy_data - clean_data
    signal_power = np.sum(clean_data ** 2)
    noise_power = np.sum(noise ** 2)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# 让代码被导入时不执行，只有在直接运行时才会执行
if __name__ == "__main__":
    clean_file = "./clean.wav"
    noisy_file = "./noisy.wav"
    snr = calculate_snr(clean_file, noisy_file)
    print ("The signal-to-noise ratio is {:.2f} dB".format(snr))



