# Solution — Challenge A: Physical AI & Robotics

## Your Name

**Aarya Yadav**

## Approach

I implemented a complete EMG signal processing and machine learning pipeline to classify five different hand gestures from four-channel EMG recordings. The pipeline follows a modular workflow consisting of data loading, signal preprocessing, window segmentation, feature extraction, model training, evaluation, and prediction on unseen data.

The implementation was designed with readability, modularity, and reproducibility in mind. Each stage of the pipeline is implemented as an independent function, making the solution easy to understand and extend.

### Signal Preprocessing

The raw EMG signals were first processed using a **4th-order Butterworth band-pass filter (20–450 Hz)** to remove low-frequency motion artifacts and high-frequency noise while preserving the useful EMG frequency band.

After filtering, the signals were normalized using **StandardScaler**, ensuring that all four channels contribute equally during model training.

### Feature Extraction

The continuous EMG signals were segmented into overlapping windows of **200 samples** with an overlap of **50 samples**.

For each window and each EMG channel, four commonly used time-domain features were extracted:

* Root Mean Square (RMS)
* Mean Absolute Value (MAV)
* Zero Crossing Rate (ZCR)
* Waveform Length (WL)

Since there are four channels, this results in **16 features per window**. These features were selected because they effectively capture muscle activation, signal amplitude, waveform complexity, and frequency-related characteristics while remaining computationally efficient.

### Classification

A **Random Forest Classifier** was selected because it performs well on structured feature-based datasets, requires minimal preprocessing, is robust to overfitting, and provides feature importance information for model interpretability.

Configuration:

* 200 decision trees
* Random state = 42
* Parallel training using all CPU cores (`n_jobs=-1`)

The dataset was split into training and testing sets using an **80:20 stratified split**.

**Final Test Accuracy:** **98.51%**

In addition to classification, the implementation also generates:

* Confusion Matrix
* Feature Importance Plot
* Prediction Confidence Scores
* Classification Report
* Prediction CSV

### Challenges & Trade-offs

One of the main challenges was selecting features that are both computationally efficient and informative enough for gesture classification. I chose time-domain features because they are widely used in EMG applications and are well suited for real-time systems.

Another challenge was assigning labels to overlapping windows. I addressed this by assigning each window the majority class label of the samples contained within it.

Random Forest was chosen instead of more complex deep learning models because it provides strong performance on this dataset while remaining fast to train, interpretable, and easier to deploy.

---

## How to Run

```bash
# Navigate to the challenge directory

cd challenges/challenge-a

# Install dependencies

pip install -r requirements.txt

# Run the solution

python src/placeholder.py
```

Running the script automatically:

* Loads the training dataset
* Preprocesses EMG signals
* Extracts features
* Trains the classifier
* Evaluates model performance
* Predicts the test dataset
* Generates plots and output files

---

## Results

The final implementation achieved:

* **Test Accuracy:** 98.51%
* Five-class EMG gesture classification
* Automatic prediction generation for unseen test data

Generated artifacts:

* `plots/raw_vs_filtered.png`
* `plots/confusion_matrix.png`
* `plots/feature_importance.png`
* `plots/feature_distributions.png`
* `predictions.csv`
* `classification_report.txt`

---

## Bonus (if applicable)

In addition to the required implementation, I added several enhancements:

* Feature importance visualization using the trained Random Forest model.
* Feature distribution visualization using a 4×4 grid of boxplots showing how all 16 features (RMS, MAV, ZCR, WL × 4 channels) are distributed across each grip type, providing insight into feature separability.
* Prediction confidence estimation using class probabilities (`predict_proba()`).
* Automatic export of predictions to `predictions.csv`.
* Automatic generation of a classification report.
* Automatic generation of signal visualization and evaluation plots.
* **Real-time simulation** that processes one EMG window at a time and reports per-window prediction latency. The simulation prints each prediction with its confidence and processing time, then outputs aggregate latency statistics (mean, median, min, max, std, 95th percentile) and a pass/fail check against a 15ms real-time target.

These additions improve the interpretability and usability of the solution while demonstrating readiness for deployment on real prosthetic hardware.
