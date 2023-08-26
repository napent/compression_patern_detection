import numpy as np


# Function to generate the accurate "Head and Shoulders" pattern
def generate_accurate_head_and_shoulders():
    shoulder = [0, 1, 2, 3, 2, 1]
    head = [2, 3, 4, 5, 6, 5, 4, 3]
    return shoulder + head + list(reversed(shoulder))


# Function to generate variable "Head and Shoulders" patterns based on frequency and amplitude multipliers
def generate_variable_head_and_shoulders(freq_multiplier=1.0, amplitude_multiplier=1.0):
    base_pattern = generate_accurate_head_and_shoulders()
    x = np.linspace(0, len(base_pattern) - 1, len(base_pattern))
    interpolated_x = np.linspace(0, len(base_pattern) - 1, int(len(base_pattern) * freq_multiplier))
    interpolated_pattern = np.interp(interpolated_x, x, base_pattern)
    return interpolated_pattern * amplitude_multiplier


def convert_to_difference(data):
    percentage_change = []
    for i in range(1, len(data)):
        if data[i - 1] != 0:
            percentage_change.append((data[i] - data[i - 1]))
        else:
            percentage_change.append(0)
    return np.array(percentage_change)


# Generate additional "Head and Shoulders" reference patterns with subtle variations
additional_patterns = {
    "Slightly Higher Frequency": generate_variable_head_and_shoulders(freq_multiplier=1.15),
    "Slightly Lower Frequency": generate_variable_head_and_shoulders(freq_multiplier=0.85),
    "Slightly Higher Amplitude": generate_variable_head_and_shoulders(amplitude_multiplier=1.2),
    "Slightly Lower Amplitude": generate_variable_head_and_shoulders(amplitude_multiplier=0.8),
    "Mixed High Freq. & High Amp.": generate_variable_head_and_shoulders(freq_multiplier=1.15,
                                                                         amplitude_multiplier=1.2),
    "Mixed Low Freq. & Low Amp.": generate_variable_head_and_shoulders(freq_multiplier=0.85,
                                                                       amplitude_multiplier=0.8)
}

# Merge the original and additional reference patterns
all_reference_patterns = {
    "Original": generate_accurate_head_and_shoulders(),
    **additional_patterns
}


def generate_synthetic_data_with_trend():
    np.random.seed(42)  # for reproducibility

    # 1. Random Walk with Drift
    drift = 0.02  # slight upward trend
    random_walk = [0]
    for i in range(1, 1000):
        movement = drift + 0.3 * np.random.randn()  # reducing the randomness to make the pattern more discernible
        value = random_walk[-1] + movement
        random_walk.append(value)

    # 2. Varying Volatility but reduced to ensure the pattern is discernible
    volatilities = 0.3 * (np.abs(np.sin(np.linspace(0, 10, 1000))) + 0.5)
    noise = np.random.randn(1000) * volatilities

    # Combine the random walk and noise
    asset_price = np.array(random_walk) + noise

    # 3. Embed the "Head and Shoulders" pattern in a predefined location
    pattern = all_reference_patterns['Original']
    embed_start = 450
    embed_end = embed_start + len(pattern)
    asset_price[embed_start:embed_end] = pattern + asset_price[embed_start:embed_end]

    # 4. Autoregressive Component
    for i in range(1, len(asset_price)):
        asset_price[i] += 0.4 * asset_price[i - 1]

    # Convert to percentage change representation
    asset_percentage_change = convert_to_difference(asset_price)
    all_reference_percentage_change = {label: convert_to_difference(pattern) for label, pattern in
                                       all_reference_patterns.items()}

    return asset_price, asset_percentage_change, all_reference_percentage_change


def generate_synthetic_data():
    # Generate synthetic data with embedded "Head and Shoulders" pattern
    np.random.seed(42)  # for reproducibility

    window_size = len(all_reference_patterns['Original'])
    pattern = all_reference_patterns['Original']

    # Generate a mild linear trend
    trend = np.linspace(0, 10, 1000)

    # Random noise
    noise = np.random.randn(1000)

    # Combine trend and noise
    asset_price = noise

    # Introduce the "Head and Shoulders" pattern
    embed_start = 450
    embed_end = embed_start + window_size
    asset_price[embed_start:embed_end] = pattern
    asset_price += trend + noise

    # Convert to percentage change representation
    asset_percentage_change = convert_to_difference(asset_price)
    all_reference_percentage_change = {label: convert_to_difference(pattern) for label, pattern in
                                       all_reference_patterns.items()}

    return asset_price, asset_percentage_change, all_reference_percentage_change

