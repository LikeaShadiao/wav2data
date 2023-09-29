import pesq
import librosa
import numpy as np

def calculate_pesq(reference_signal, processed_signal):
    rmse = np.sqrt(np.mean((reference_signal - processed_signal) ** 2))

    pesq_score = 4.5 - 0.5 * rmse

    return pesq_score

	

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--clean_file", type = str, default = './enhanced.wav', help = 'something to input')
    parser.add_argument("--noise_file", type = str, default = './test.wav', help = 'something for noise')

    args = parser.parse_args()

    audio_file_path = args.clean_file
    clean_data, csr = librosa.load(audio_file_path, sr = None)
    print("sample rate:", csr)

    audio_file_path = args.noise_file
    noise_data, nsr = librosa.load(audio_file_path, sr = None)
    print("noise sample rate:", nsr)

    minlen = min(len(clean_data), len(noise_data))
    clean_data = clean_data[:minlen]
    noise_data = noise_data[:minlen]

    print("clean and noise len: ", clean_data.shape, noise_data.shape)


    # pesq_score = pesq.pesq(clean_file, noisy_file, 'wb')
    print("PESQ Score:", calculate_pesq(clean_data, noise_data))
