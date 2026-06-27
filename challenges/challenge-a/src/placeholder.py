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
        signals: raw signal array (n_samples, 4)
        fs: sampling frequency in Hz
    
    Returns:
        filtered_signals: preprocessed signal array
    """
    # TODO: Apply bandpass filter (20-450 Hz)
    # TODO: Normalize signals
    pass


def segment_windows(signals, labels=None, window_size=200, overlap=50):
    """
    Segment continuous signal into overlapping windows.
    
    Args:
        signals: preprocessed signal array (n_samples, 4)
        labels: optional label array
        window_size: samples per window (200 = 200ms at 1000Hz)
        overlap: overlap in samples
    
    Returns:
        windows: array of shape (n_windows, window_size, 4)
        window_labels: array of labels per window (if labels provided)
    """
    # TODO: Implement windowing with overlap
    pass


def extract_features(windows):
    """
    Extract features from each signal window.
    
    Args:
        windows: array of shape (n_windows, window_size, 4)
    
    Returns:
        features: array of shape (n_windows, n_features)
    """
    # TODO: Extract RMS, MAV, Zero Crossing Rate, Waveform Length
    # for each of the 4 channels
    pass


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
    # filtered = preprocess(signals)
    
    # Step 3: Segment into windows
    print("Segmenting into windows...")
    # windows, window_labels = segment_windows(filtered, labels)
    
    # Step 4: Extract features
    print("Extracting features...")
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
