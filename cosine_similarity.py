import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _compute_cosine_similarity(x, y):
    """
    Compute cosine similarity between two byte sequences x and y by treating them as vectors.
    Pad the shorter vector with zeros to match the length of the longer one.
    """
    # Convert byte sequences to integer vectors
    x_vector = np.array([int(b) for b in x])
    y_vector = np.array([int(b) for b in y])

    # Padding
    length_diff = len(x_vector) - len(y_vector)
    # Padding the shorter vector with zeros to match the length of the longer one
    if length_diff > 0:
        y_vector = np.pad(y_vector, (0, length_diff), 'constant')
    elif length_diff < 0:
        x_vector = np.pad(x_vector, (0, -length_diff), 'constant')

    # Reshape vectors to 2D arrays for cosine_similarity function
    x_vector = x_vector.reshape(1, -1)
    y_vector = y_vector.reshape(1, -1)

    return cosine_similarity(x_vector, y_vector)[0][0]


def compute_cosine_similarity(asset_percent_change, references_percentage_change, compress_function):
    cosine_similarities = []
    window_size = len(references_percentage_change['Original']) - 1

    for i in range(len(asset_percent_change) - window_size + 1):
        asset_window = asset_percent_change[i:i + window_size]
        compressed_window = compress_function.compress(asset_window.tobytes())

        max_similarity = float('-inf')  # start with negative infinity so any real number will be greater

        for ref_movement in references_percentage_change.values():
            similarity = _compute_cosine_similarity(compressed_window,
                                                    compress_function.compress(ref_movement.tobytes()))
            if similarity > max_similarity:
                max_similarity = similarity

        cosine_similarities.append(max_similarity)

    return cosine_similarities
