"""
Kawatek Internship Challenge A: EMG Signal Classification Pipeline
==================================================================

Your task is to build a signal processing and classification pipeline
that converts raw EMG signals into grip pattern predictions.

Start by implementing the functions below, then add your own logic.

Good luck!
"""

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(filepath: str):
    """
    Load EMG signal data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        timestamps: NumPy array of timestamps.
        signals: NumPy array of shape (n_samples, 4).
        labels: NumPy array of labels if present, otherwise None.
    """

    df = pd.read_csv(filepath)
    timestamps = df["timestamp"].to_numpy()
    signals = df[["ch1", "ch2", "ch3", "ch4"]].to_numpy(dtype=np.float64, copy=False)
    labels = df["label"].to_numpy() if "label" in df.columns else None
    return timestamps, signals, labels


def preprocess(signals, fs=1000):
    """
    Apply preprocessing to raw EMG signals.

    Args:
        signals: Raw EMG signal array (n_samples, 4)
        fs: Sampling frequency in Hz

    Returns:
        normalized:
            Filtered and normalized signals for machine learning.

        filtered_only:
            Filtered signals before normalization for visualization.
    """

    nyquist = fs / 2

    low = 20 / nyquist
    high = 450 / nyquist

    b, a = butter(
        N=4,
        Wn=[low, high],
        btype="bandpass"
    )

    filtered = np.zeros_like(signals)

    for channel in range(signals.shape[1]):
        filtered[:, channel] = filtfilt(
            b,
            a,
            signals[:, channel]
        )

    filtered_only = filtered.copy()

    scaler = StandardScaler()

    normalized = scaler.fit_transform(filtered)

    return normalized, filtered_only


def plot_signals(timestamps, raw_signals, filtered_signals):
    """
    Plot raw and filtered EMG signals for all four channels.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True
    )

    channel_names = [
        "EMG Channel 1",
        "EMG Channel 2",
        "EMG Channel 3",
        "EMG Channel 4"
    ]

    for i, ax in enumerate(axes.flat):

        ax.plot(
            timestamps[:1000],
            raw_signals[:1000, i],
            label="Raw",
            alpha=0.6
        )

        ax.plot(
            timestamps[:1000],
            filtered_signals[:1000, i],
            label="Filtered",
            linewidth=1.8
        )

        ax.set_title(channel_names[i])
        ax.grid(True)

    axes[1,0].set_xlabel("Time (seconds)")
    axes[1,1].set_xlabel("Time (seconds)")

    axes[0,0].set_ylabel("Amplitude")
    axes[1,0].set_ylabel("Amplitude")

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(
        "plots/raw_vs_filtered.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def segment_windows(signals, labels=None, window_size=200, overlap=50):
    """
    Segment continuous EMG signals into overlapping windows.

    Args:
        signals: Preprocessed signal array (n_samples, 4)
        labels: Optional label array
        window_size: Number of samples per window
        overlap: Number of overlapping samples

    Returns:
        windows:
            NumPy array of shape
            (n_windows, window_size, 4)

        window_labels:
            Label for each window
            (None for test data)
    """

    step_size = window_size - overlap

    windows = []
    window_labels = []

    for start_index in range(
        0,
        len(signals) - window_size + 1,
        step_size
    ):

        end_index = start_index + window_size

        window = signals[start_index:end_index]

        windows.append(window)

        if labels is not None:

            label_window = labels[start_index:end_index]

            values, counts = np.unique(
                label_window,
                return_counts=True
            )

            majority_label = values[np.argmax(counts)]

            window_labels.append(majority_label)

    windows = np.array(windows)

    if labels is not None:
        window_labels = np.array(window_labels)
    else:
        window_labels = None

    return windows, window_labels

def rms(signal):
    """
    Compute Root Mean Square (RMS) of a signal.
    """
    return np.sqrt(np.mean(signal ** 2))

def mav(signal):
    """
    Compute Mean Absolute Value (MAV) of a signal.
    """
    return np.mean(np.abs(signal))

def zero_crossing_rate(signal):
    """
    Compute Zero Crossing Rate (ZCR) of a signal.
    """
    return np.sum(np.diff(np.sign(signal)) != 0)

def waveform_length(signal):
    """
    Compute Waveform Length (WL) of a signal.
    """
    return np.sum(np.abs(np.diff(signal)))


def extract_features(windows):
    """
    Extract time-domain features from each EMG window.

    Args:
        windows:
            Array of shape
            (n_windows, window_size, 4)

    Returns:
        Feature matrix of shape
        (n_windows, 16)
    """

    feature_matrix = []

    for window in windows:

        window_features = []

        for channel in range(window.shape[1]):

            signal = window[:, channel]

            window_features.extend([
                rms(signal),
                mav(signal),
                zero_crossing_rate(signal),
                waveform_length(signal)
            ])

        feature_matrix.append(window_features)

    return np.array(feature_matrix)


def train_classifier(features, labels):
    """
    Train a classifier on extracted features.
    
    Args:
        features: feature array (n_samples, n_features)
        labels: grip label array
    
    Returns:
        model: trained classifier
        accuracy: test accuracy score
    """
    # TODO: Train/test split, train model, evaluate
    pass


def predict(model, features):
    """
    Predict grip patterns from features.
    
    Args:
        model: trained classifier
        features: feature array for test data
    
    Returns:
        predictions: array of predicted grip labels
    """
    # TODO: Generate predictions
    pass


if __name__ == "__main__":
    # Main pipeline
    print("=== Kawatek EMG Classification Pipeline ===\n")
    
    # Step 1: Load training data
    print("Loading training data...")

    timestamps, signals, labels = load_data("data/emg_signals.csv")

    print(f"Number of samples : {len(signals)}")
    print(f"Signal shape      : {signals.shape}")
    print(f"Timestamp shape   : {timestamps.shape}")
    print(f"Unique labels     : {np.unique(labels)}")
    
    # Step 2: Preprocess
    print("Preprocessing signals...")

    normalized, filtered = preprocess(signals)

    print("Filtering complete.")
    print(f"Filtered shape : {filtered.shape}")

    print("Generating preprocessing visualization...")
    plot_signals(
        timestamps,
        signals,
        filtered
    )
    # filtered = preprocess(signals)
    
    # Step 3: Segment into windows
    print("Segmenting into windows...")

    windows, window_labels = segment_windows(
        normalized,
        labels
    )

    print(f"Number of windows : {len(windows)}")
    print(f"Window shape      : {windows.shape}")

    print()

    print("First 10 window labels:")

    print(window_labels[:10])
    # windows, window_labels = segment_windows(filtered, labels)
    
    # Step 4: Extract features
    print("Extracting features...")

    features = extract_features(windows)
    # features = extract_features(windows)
    
    # Step 5: Train classifier
    print("Training classifier...")
    # model, accuracy = train_classifier(features, window_labels)
    # print(f"Training accuracy: {accuracy:.2%}")
    
    # Step 6: Predict on test data
    print("Predicting test data...")
    # test_timestamps, test_signals, _ = load_data("../data/test_signals.csv")
    # test_filtered = preprocess(test_signals)
    # test_windows, _ = segment_windows(test_filtered)
    # test_features = extract_features(test_windows)
    # predictions = predict(model, test_features)
    
    # Step 7: Output predictions
    # for i, pred in enumerate(predictions):
    #     print(f"Window {i:03d}: {pred}")
    
    print("\nDone! Don't forget to write your SOLUTION.md")
