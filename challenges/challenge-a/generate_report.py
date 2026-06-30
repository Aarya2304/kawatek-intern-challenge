"""
Kawatek Challenge A — HTML Report Generator
=============================================

Generates a self-contained dark-themed HTML report with embedded
base64 plots that match the pipeline in src/placeholder.py.

Usage:
    python generate_report.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import base64
from io import BytesIO

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")

FEATURE_NAMES = [
    "CH1 RMS", "CH1 MAV", "CH1 ZCR", "CH1 WL",
    "CH2 RMS", "CH2 MAV", "CH2 ZCR", "CH2 WL",
    "CH3 RMS", "CH3 MAV", "CH3 ZCR", "CH3 WL",
    "CH4 RMS", "CH4 MAV", "CH4 ZCR", "CH4 WL",
]

# ── Helpers ──────────────────────────────────────────────────────────


def get_b64_plot():
    """Save the current matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def bandpass_filter(data, fs=1000):
    """Apply a 4th-order Butterworth bandpass filter (20-450 Hz)."""
    nyquist = fs / 2
    low = 20 / nyquist
    high = 450 / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="bandpass")
    return filtfilt(b, a, data)


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def mav(signal):
    return np.mean(np.abs(signal))

def zcr(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)

def wl(signal):
    return np.sum(np.abs(np.diff(signal)))


def extract_window_features(window):
    """Extract 16 features (4 per channel) from a single window."""
    features = []
    for ch in range(window.shape[1]):
        s = window[:, ch]
        features.extend([rms(s), mav(s), zcr(s), wl(s)])
    return features


# ── 1. Load Data ─────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv("data/emg_signals.csv")
timestamps = df["timestamp"].to_numpy()
signals = df[["ch1", "ch2", "ch3", "ch4"]].to_numpy(dtype=np.float64)
labels = df["label"].to_numpy()
print(f"Data loaded: {df.shape}")

# ── 2. Preprocess ────────────────────────────────────────────────────

print("Preprocessing...")
fs = 1000
filtered = np.zeros_like(signals)
for ch in range(4):
    filtered[:, ch] = bandpass_filter(signals[:, ch], fs)

filtered_for_viz = filtered.copy()

scaler = StandardScaler()
normalized = scaler.fit_transform(filtered)

# ── 3. Plot: Raw vs Filtered (all 4 channels) ───────────────────────

print("Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
ch_names = ["EMG Channel 1", "EMG Channel 2", "EMG Channel 3", "EMG Channel 4"]
for i, ax in enumerate(axes.flat):
    ax.plot(timestamps[:1000], signals[:1000, i], label="Raw", alpha=0.6)
    ax.plot(timestamps[:1000], filtered_for_viz[:1000, i], label="Filtered", linewidth=1.8)
    ax.set_title(ch_names[i])
    ax.grid(True, alpha=0.3)
axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 1].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("Amplitude")
axes[1, 0].set_ylabel("Amplitude")
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper right")
plt.tight_layout(rect=[0, 0, 1, 0.97])
img_raw_filtered = get_b64_plot()

# ── 4. Segment Windows (200 samples, 50 overlap) ────────────────────

window_size = 200
overlap = 50
step = window_size - overlap

windows = []
window_labels = []
for start in range(0, len(normalized) - window_size + 1, step):
    end = start + window_size
    windows.append(normalized[start:end])
    lbl_window = labels[start:end]
    vals, counts = np.unique(lbl_window, return_counts=True)
    window_labels.append(vals[np.argmax(counts)])

windows = np.array(windows)
window_labels = np.array(window_labels)

# ── 5. Extract Features ─────────────────────────────────────────────

features_list = [extract_window_features(w) for w in windows]
X = np.array(features_list)
y = window_labels
print(f"Features extracted: {X.shape}")

# ── 6. Plot: Feature Distributions (4x4 grid) ───────────────────────

feature_df = pd.DataFrame(X, columns=FEATURE_NAMES)
feature_df["Grip Type"] = y

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle("Feature Distributions per Grip Type", fontsize=16, fontweight="bold")
grip_types = sorted(feature_df["Grip Type"].unique())
colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8"]

for i, (ax, fname) in enumerate(zip(axes.flat, FEATURE_NAMES)):
    grip_data = [feature_df[feature_df["Grip Type"] == g][fname].values for g in grip_types]
    bp = ax.boxplot(grip_data, tick_labels=grip_types, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(fname, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
img_feat_dist = get_b64_plot()

# ── 7. Train Classifier (200 trees, stratified) ─────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report_text = classification_report(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")

# ── 8. Plot: Confusion Matrix ───────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
img_cm = get_b64_plot()

# ── 9. Plot: Feature Importance ──────────────────────────────────────

importance = clf.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
bars = plt.bar(np.array(FEATURE_NAMES)[sorted_idx], importance[sorted_idx])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.xlabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Extracted Features")
plt.title("Feature Importance (Random Forest)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
img_feat_imp = get_b64_plot()

# ── 10. Real-time Simulation ────────────────────────────────────────

# Load and preprocess test data
test_df = pd.read_csv("data/test_signals.csv")
test_signals = test_df[["ch1", "ch2", "ch3", "ch4"]].to_numpy(dtype=np.float64)
test_filtered = np.zeros_like(test_signals)
for ch in range(4):
    test_filtered[:, ch] = bandpass_filter(test_signals[:, ch], fs)
test_normalized = scaler.transform(test_filtered)

n_test_windows = (len(test_normalized) - window_size) // step + 1
latencies = []
for i in range(n_test_windows):
    s = i * step
    e = s + window_size
    window = test_normalized[s:e]
    t0 = time.perf_counter()
    feats = np.array(extract_window_features(window)).reshape(1, -1)
    clf.predict(feats)
    clf.predict_proba(feats)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies = np.array(latencies)
mean_lat = latencies.mean()
median_lat = np.median(latencies)
min_lat = latencies.min()
max_lat = latencies.max()
p95_lat = np.percentile(latencies, 95)

# ── 11. Generate HTML Report ────────────────────────────────────────

html_content = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kawatek Challenge A: EMG Classification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0f172a;
            color: #cbd5e1;
        }}
        .glass-card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.15);
        }}
        .glow-text {{
            color: #fff;
            text-shadow: 0 0 20px rgba(56, 189, 248, 0.6);
        }}
        .gradient-text {{
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .plot-container {{
            background: #000;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
            border-radius: 0.375rem;
        }}
        .stat-value {{
            font-variant-numeric: tabular-nums;
        }}
    </style>
</head>
<body class="relative min-h-screen overflow-x-hidden selection:bg-sky-500/30">
    <!-- Background glowing orbs -->
    <div class="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-600/20 rounded-full mix-blend-screen filter blur-[100px] animate-pulse pointer-events-none"></div>
    <div class="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-indigo-600/20 rounded-full mix-blend-screen filter blur-[100px] animate-pulse pointer-events-none" style="animation-delay: 2s;"></div>

    <div class="max-w-4xl mx-auto px-6 py-12 relative z-10">

        <header class="mb-16 text-center">
            <div class="inline-block px-4 py-1.5 rounded-full border border-sky-500/30 bg-sky-500/10 text-sky-400 text-sm font-semibold tracking-wide mb-6">
                Kawatek 2027 Internship
            </div>
            <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
                <span class="glow-text">Physical AI &amp; Robotics</span>
            </h1>
            <p class="text-xl text-slate-400 font-medium max-w-2xl mx-auto">
                Real-time processing pipeline for the RYO&reg; bionic hand &mdash; Decoding human intention from electromyography signals.
            </p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="glass-card p-6 text-center">
                <div class="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Model Accuracy</div>
                <div class="text-4xl font-bold gradient-text">{acc*100:.1f}%</div>
            </div>
            <div class="glass-card p-6 text-center">
                <div class="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Algorithm</div>
                <div class="text-2xl font-bold text-white mt-1">Random Forest</div>
                <div class="text-sm text-slate-500 mt-1">200 trees</div>
            </div>
            <div class="glass-card p-6 text-center">
                <div class="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Avg Latency</div>
                <div class="text-2xl font-bold text-white mt-1 stat-value">{mean_lat:.1f}ms</div>
            </div>
            <div class="glass-card p-6 text-center">
                <div class="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Features</div>
                <div class="text-2xl font-bold text-white mt-1">16</div>
                <div class="text-sm text-slate-500 mt-1">4 per channel</div>
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                <svg class="w-6 h-6 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path></svg>
                Project Architecture
            </h2>
            <div class="space-y-4 text-slate-300 leading-relaxed">
                <p>This report details the implementation of a real-time signal classification pipeline that converts raw sensor data into motor commands. The architecture consists of four stages:</p>
                <ul class="list-disc list-inside space-y-2 ml-2">
                    <li><strong class="text-sky-300">Signal Preprocessing:</strong> A 4th-order Butterworth bandpass filter (20&ndash;450 Hz) removes environmental noise and motion artifacts. Signals are then normalized with StandardScaler.</li>
                    <li><strong class="text-sky-300">Window Segmentation:</strong> The continuous datastream is segmented into 200-sample windows with 50-sample overlap. Each window is assigned the majority class label.</li>
                    <li><strong class="text-sky-300">Feature Engineering:</strong> Four time-domain features (RMS, MAV, ZCR, WL) are extracted per channel, yielding 16 features per window.</li>
                    <li><strong class="text-sky-300">Classification:</strong> A Random Forest classifier (200 trees, stratified 80/20 split) identifies one of five grip patterns: <code class="bg-slate-700/50 px-2 py-0.5 rounded text-sky-200">rest</code>, <code class="bg-slate-700/50 px-2 py-0.5 rounded text-sky-200">power_grip</code>, <code class="bg-slate-700/50 px-2 py-0.5 rounded text-sky-200">pinch</code>, <code class="bg-slate-700/50 px-2 py-0.5 rounded text-sky-200">lateral</code>, and <code class="bg-slate-700/50 px-2 py-0.5 rounded text-sky-200">point</code>.</li>
                </ul>
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <svg class="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path></svg>
                Signal Filtering
            </h2>
            <p class="text-slate-400">Raw vs filtered EMG signals across all four channels (first 1000 samples).</p>
            <div class="plot-container">
                <img src="data:image/png;base64,{img_raw_filtered}" alt="Raw vs Filtered Signal">
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <svg class="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"></path></svg>
                Feature Distributions
            </h2>
            <p class="text-slate-400">Boxplots of all 16 time-domain features (RMS, MAV, ZCR, WL &times; 4 channels) across the five grip types.</p>
            <div class="plot-container">
                <img src="data:image/png;base64,{img_feat_dist}" alt="Feature Distributions">
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <svg class="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                Feature Importance
            </h2>
            <p class="text-slate-400">Relative importance of each extracted feature as determined by the Random Forest model.</p>
            <div class="plot-container">
                <img src="data:image/png;base64,{img_feat_imp}" alt="Feature Importance">
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <svg class="w-6 h-6 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                Classification Performance
            </h2>
            <p class="text-slate-400 mb-6">Confusion matrix of predictions against the held-out test set.</p>
            <div class="plot-container" style="max-width: 600px; margin-left: auto; margin-right: auto;">
                <img src="data:image/png;base64,{img_cm}" alt="Confusion Matrix">
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                <svg class="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                Real-time Simulation
            </h2>
            <p class="text-slate-400 mb-6">Per-window latency metrics from processing {n_test_windows} test windows sequentially.</p>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">Mean Latency</div>
                    <div class="text-xl font-bold text-white stat-value">{mean_lat:.2f} ms</div>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">Median Latency</div>
                    <div class="text-xl font-bold text-white stat-value">{median_lat:.2f} ms</div>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">Min Latency</div>
                    <div class="text-xl font-bold text-white stat-value">{min_lat:.2f} ms</div>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">Max Latency</div>
                    <div class="text-xl font-bold text-white stat-value">{max_lat:.2f} ms</div>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">95th Percentile</div>
                    <div class="text-xl font-bold text-white stat-value">{p95_lat:.2f} ms</div>
                </div>
                <div class="bg-slate-800/50 rounded-lg p-4 text-center">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">Windows Processed</div>
                    <div class="text-xl font-bold text-white stat-value">{n_test_windows}</div>
                </div>
            </div>
        </div>

        <div class="glass-card p-8 mb-8">
            <h2 class="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                <svg class="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
                Classification Report
            </h2>
            <pre class="bg-slate-900/80 rounded-lg p-4 text-sm text-slate-300 overflow-x-auto font-mono">{report_text}</pre>
        </div>

        <footer class="text-center text-slate-500 text-sm mt-12 pb-8">
            &copy; 2027 Kawatek Co., Ltd. &mdash; Intelligent Bionics &amp; Human Augmentation
        </footer>
    </div>
</body>
</html>
"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Report successfully generated at report.html")
