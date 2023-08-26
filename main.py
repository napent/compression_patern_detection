import matplotlib.pyplot as plt
import zlib
import bz2
import gzip

from compression_ratio import compression_ratio_similarity
from cosine_similarity import compute_cosine_similarity
from normalized_compression_distance import normalized_compression_distance
from synthetic_data import generate_synthetic_data, generate_synthetic_data_with_trend

asset_price, asset_percentage_change, all_reference_percentage_change = generate_synthetic_data()

# select compression algorithm
selected_compression = bz2
selected_compression = zlib
selected_compression = gzip

cosine_similarities = compute_cosine_similarity(asset_percentage_change, all_reference_percentage_change,
                                                selected_compression)

compression_ratios = compression_ratio_similarity(asset_percentage_change, all_reference_percentage_change,
                                                  selected_compression)

NCD = normalized_compression_distance(asset_percentage_change,
                                      all_reference_percentage_change,
                                      selected_compression)

window_size = len(all_reference_percentage_change['Original']) - 1


plt.figure(figsize=(15, 12))

# Subplot 1: Synthetic close price data with NCD-based detection highlighted
plt.subplot(4, 1, 1)
plt.plot(asset_price, label="Synthetic Close Price", color='blue')
threshold = 0.75
for i, ncd in enumerate(NCD):
    if ncd < threshold:
        plt.axvspan(i + 1, i + window_size + 1, color='yellow', alpha=0.4)  # +1 due to percentage change representation
plt.title("Synthetic Data with Detected Patterns Highlighted (NCD-based)")
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Subplot 2: NCD values over time for percentage change movements with threshold lines
plt.subplot(4, 1, 2)
plt.plot(NCD, label="NCD with Reference (Percentage Change)")
plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold (NCD={threshold:.2f})")
plt.title("NCD Values for Percentage Change Asset Movements with 'Head and Shoulders' Reference")
plt.xlabel("Time (days)")
plt.ylabel("NCD Value")
plt.legend()
plt.grid(True)

# Subplot 3: Compression ratios over time
plt.subplot(4, 1, 3)
plt.plot(compression_ratios, label="Compression Ratios (Percentage Change)", color='green')
plt.title("Compression Ratios Over Time for Percentage Change Asset Movements")
plt.xlabel("Time (days)")
plt.ylabel("Compression Ratio")
plt.legend()
plt.grid(True)

# Subplot 2: Cosine similarity values over time for percentage change movements
plt.subplot(4, 1, 4)
plt.plot(cosine_similarities, label="Cosine Similarity with a Reference (Percentage Change)")
plt.title("Cosine Similarity Values for Percentage Change Asset Movements with 'Head and Shoulders' Reference")
plt.xlabel("Time (days)")
plt.ylabel("Cosine Similarity Value")
plt.legend()
plt.grid(True)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
