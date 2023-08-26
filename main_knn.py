import gzip

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from synthetic_data import generate_synthetic_data, generate_synthetic_data_with_trend


# ------------------------ Functions ------------------------

def compute_NCD(x, y, selected_compression):
    compress_fn = lambda data: len(selected_compression.compress(data))
    xz = compress_fn(x)
    yz = compress_fn(y)
    x_yz = compress_fn(x + y)
    ncd = (x_yz - min(xz, yz)) / max(xz, yz)
    return ncd


# ------------------------ Data Preparation ------------------------

# Generate synthetic data
asset_price, asset_percentage_change, all_reference_percentage_change = generate_synthetic_data()

# Compute NCD values for training
labels = list(all_reference_percentage_change.keys())
training_data = []

for pattern1 in all_reference_percentage_change.values():
    row = []
    for pattern2 in all_reference_percentage_change.values():
        row.append(compute_NCD(pattern1.tobytes(), pattern2.tobytes(), gzip))
    training_data.append(row)
window_size = len(next(iter(all_reference_percentage_change.values())))

# Consolidated code for pattern detection using distance-weighted KNN and visualization

# Train KNN Classifier with k=3 and distance-based weighting
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(training_data, labels)

# Get probability estimates for each window of data using the weighted classifier
probability_estimates_weighted = []

for i in range(len(asset_percentage_change) - window_size + 1):
    window = asset_percentage_change[i:i + window_size].tobytes()
    distances = [compute_NCD(window, pattern.tobytes(), gzip) for pattern in all_reference_percentage_change.values()]
    probs = knn.predict_proba([distances])[0]
    probability_estimates_weighted.append(probs)

probability_estimates_weighted = np.array(probability_estimates_weighted)
# ------------------------ Data Preparation ------------------------
threshold = 0.156
# Compute the maximum probability across all reference patterns for each window
max_probabilities = np.max(probability_estimates_weighted, axis=1)
# Get the label of the detected pattern based on maximum probability
detected_labels = []
for probs in probability_estimates_weighted:
    if max(probs) > threshold:
        detected_label = labels[np.argmax(probs)]
        detected_labels.append(detected_label)
    else:
        detected_labels.append("unclassified")

# ------------------------ Data Preparation ------------------------

# Determine the x-axis limits for consistency across plots
x_min = 0
x_max = len(asset_price)

# Visualization with consistent x-axis limits
fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

# Plot the asset price with detected patterns highlighted based on max probability
axes[0].plot(asset_price, label='Asset Price', color='blue')
for i, prob in enumerate(max_probabilities):
    if prob > threshold:  # If max probability is greater than 0.5, highlight the pattern
        axes[0].axvspan(i, i + window_size, color='red', alpha=0.2)
axes[0].set_xlim(x_min, x_max)
axes[0].set_title('Asset Price with Detected Patterns')
axes[0].set_ylabel('Price')
axes[0].legend()

# Plot probability estimates for all reference patterns on a single subplot
for i, label in enumerate(labels):
    axes[1].plot(range(window_size + 1, len(asset_price) + 1), probability_estimates_weighted[:, i], label=f'{label}')
axes[1].set_xlim(x_min, x_max)
axes[1].set_title('Probability Estimates for Each Reference vs. Time (Weighted KNN)')
axes[1].set_ylabel('Probability')
axes[1].legend()

# Plot the maximum probability across all references for each window and annotate with detected label
axes[2].plot(range(window_size + 1, len(asset_price) + 1), max_probabilities, label='Max Probability', color='green')
for i, (prob, label) in enumerate(zip(max_probabilities, detected_labels)):
    if prob > threshold:  # If max probability is significant, annotate with label
        axes[2].annotate(label, (i + window_size, prob), fontsize=8, alpha=0.7, ha='center')
axes[2].set_xlim(x_min, x_max)
axes[2].set_title('Maximum Probability vs. Time (Weighted KNN)')
axes[2].set_ylabel('Probability')
axes[2].legend()

# Plot the label of the detected pattern
axes[3].scatter(range(window_size + 1, len(asset_price) + 1), detected_labels, label='Detected Pattern', color='blue',
                marker='o')
axes[3].set_xlim(x_min, x_max)
axes[3].set_title('Detected Pattern vs. Time')
axes[3].set_ylabel('Pattern Label')
axes[3].set_xlabel('Time')
axes[3].legend()

plt.tight_layout()
plt.show()
