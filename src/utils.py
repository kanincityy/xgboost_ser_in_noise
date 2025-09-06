import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def parse_ravdess_filename(filename, emotions_map):
    """Parses a RAVDESS filename to extract relevant metadata."""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        if len(parts) != 7:
            return None

        modality = parts[0]
        vocal_channel_code = parts[1]
        emotion_code = parts[2]
        actor_code = parts[6]

        # Basic filtering for audio-only speech and selected emotions
        if modality != "03" or vocal_channel_code != "01" or emotion_code not in emotions_map:
            return None

        return {
            "filename": filename,
            "speaker_id": int(actor_code),
            "emotion_label": emotions_map.get(emotion_code),
        }
    except Exception:
        return None

def add_noise(clean_signal, snr_db):
    """Adds white noise to a signal at a specified SNR."""
    noise = np.random.randn(len(clean_signal))
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)

    if noise_power < 1e-10: noise_power = 1e-10
    if signal_power < 1e-10: return noise * 0.001

    k = np.sqrt(signal_power / (noise_power * (10**(snr_db / 10))))
    noisy_signal = clean_signal + (k * noise)
    return noisy_signal.astype(np.float32)

def plot_audio_ser(audio_signal, sample_rate, n_fft, hop_length, n_mels, n_mfcc, plot_title, save_path=None):
    """Plots waveform, Mel spectrogram, and MFCCs."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(plot_title, fontsize=16)

    # Waveform
    librosa.display.waveshow(y=audio_signal, sr=sample_rate, ax=axs[0], alpha=0.7)
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.5)

    # Mel Spectrogram
    S_mel = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
    librosa.display.specshow(S_db_mel, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length, ax=axs[1])
    axs[1].set_title(f"Mel Spectrogram ({n_mels} bins)")
    axs[1].set_ylabel("Frequency (Mel)")

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    img_mfcc = librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, hop_length=hop_length, ax=axs[2])
    axs[2].set_title(f"MFCCs ({n_mfcc} coefficients)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("MFCC Index")
    fig.colorbar(img_mfcc, ax=axs[2], format='%+2.0f dB')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    plt.close(fig)