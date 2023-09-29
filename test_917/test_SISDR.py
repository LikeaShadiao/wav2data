import numpy as np
import librosa


def compute_si_sdr(clean_signal, denoised_signal):
	error_signal = clean_signal - denoised_signal
	alpha = np.dot(clean_signal, error_signal) / np.dot(error_signal, error_signal)
	clean_estimated_signal = alpha * denoised_signal

	si_sdr = 10 * np.log10(np.sum(clean_signal ** 2) / np.sum((clean_signal - clean_estimated_signal) ** 2))
	return si_sdr

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--clean_file", type = str, default = './enhanced.wav', help = 'something to input')
    parser.add_argument("--noise_file", type = str, default = './test.wav', help = 'something for noise')

    args = parser.parse_args()

    audio_file_path = args.clean_file
    clean_data, csr = librosa.load(audio_file_path, sr = None)
    print("cleansample rate:", csr)

    audio_file_path = args.noise_file
    noise_data, nsr = librosa.load(audio_file_path, sr = None)
    print("noise sample rate:", nsr)

    minlen = min(len(clean_data), len(noise_data))
    clean_data = clean_data[:minlen]
    noise_data = noise_data[:minlen]

    print("clean and noise len: ", clean_data.shape, noise_data.shape)


    # pesq_score = pesq.pesq(clean_file, noisy_file, 'wb')
    print("SI-SDR Score:", compute_si_sdr(clean_data, noise_data))