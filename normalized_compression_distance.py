# Compute NCD using gzip compression
def compute_NCD(x, y, selected_compression):
    compress_fn = lambda data: len(selected_compression.compress(data))
    xz = compress_fn(x)
    yz = compress_fn(y)
    x_yz = compress_fn(x + y)
    ncd = (x_yz - min(xz, yz)) / max(xz, yz)
    return ncd


def normalized_compression_distance(asset_percent_change, references_percentage_change, selected_compression):
    window_size = len(references_percentage_change['Original']) - 1  # -1 due to percentage change representation

    ncd_output = []

    for i in range(len(asset_percent_change) - window_size + 1):
        asset_window = asset_percent_change[i:i + window_size]
        min_ncd = min([compute_NCD(asset_window.tobytes(), ref_movement.tobytes(), selected_compression)
                       for ref_movement in references_percentage_change.values()])
        ncd_output.append(min_ncd)

    return ncd_output
