import numpy as np


def compression_ratio_similarity(asset_percent_change, references_percentage_change, selected_compression):
    window_size = len(references_percentage_change['Original'])

    # Compute the compression ratio for each window
    compression_ratios = []
    for i in range(len(asset_percent_change) - window_size + 1):
        window_movement = asset_percent_change[i:i + window_size]

        # Concatenate window_movement with references_percentage_change['Original']
        combined_data = np.concatenate([window_movement, references_percentage_change['Original']])

        compressed_size = len(selected_compression.compress(combined_data.tobytes()))
        compression_ratios.append(compressed_size / combined_data.nbytes)

    return compression_ratios
