import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def smooth_values(values, alpha):
    """
    Smooth a list of values using exponential moving average.
    :param values: List of values to smooth.
    :param alpha: Smoothing factor (0 < alpha <= 1).
    :return: Smoothed values as a NumPy array.
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# Function to load JSON data and preprocess
def load_and_process_json(file_path, scale=1.0, offset=0, min_step=None, max_step=None):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Support both possible keys
    key = "episode_reward" if "episode_reward" in data else "return_mean"
    if key not in data:
        raise ValueError(f"Neither 'episode_reward' nor 'return_mean' found in {file_path}")

    # Extract steps and values
    steps = data[key]["steps"]
    values = data[key]["values"]

    # Scale and offset steps
    scaled_steps = [(step * scale) + offset for step in steps]

    # Apply step range filtering
    filtered_steps, filtered_values = [], []
    for step, value in zip(scaled_steps, values):
        if (min_step is None or step >= min_step) and (max_step is None or step <= max_step):
            filtered_steps.append(step)
            filtered_values.append(value)

    return filtered_steps, filtered_values


# Function to plot data
def save_plot(data, labels, output_file, alpha=0.2):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    for steps, values, label in data:
        if alpha >= 1:
            plt.plot(steps, values, label=label)
        else:
            # Get next color from color cycle
            color = next(plt.gca()._get_lines.prop_cycler)["color"]

            # Plot shadow for unsmoothed values
            plt.plot(steps, values, alpha=0.2, color=color)

            # Plot smoothed curve
            smoothed_values = smooth_values(values, alpha=alpha)
            plt.plot(steps, smoothed_values, label=label, color=color)

    plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.ylim(bottom=0)
    # plt.title("")
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


# Main function
def main(file_configs, output_file):
    data = []
    for config in file_configs:
        file_path = config["file"]
        scale = config.get("scale", 1.0)
        offset = config.get("offset", 0.0)
        min_step = config.get("min_step", None)
        max_step = config.get("max_step", None)
        label = config.get("label", file_path)

        # Load and process JSON
        steps, values = load_and_process_json(file_path, scale, offset, min_step, max_step)
        data.append((steps, values, label))

    # Save the plot
    save_plot(data, [config["label"] for config in file_configs], output_file)


def parse_cli_args():
    # Argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Plot JSON data from multiple files.")
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="List of JSON files to load, with optional scale and offset parameters.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot.png",
        help="Output file name for the saved graph (e.g., 'plot.png').",
    )

    args = parser.parse_args()

    # Example file_configs for manual testing
    # Example format for argument: file.json:scale=1.0:offset=0:label="File 1"
    # Example usage: python script.py --files file1.json:scale=1.5:offset=10000 file2.json:label="File 2"
    file_configs = []
    for file_arg in args.files:
        parts = file_arg.split(":")
        file_path = parts[0]
        config = {"file": file_path}

        for part in parts[1:]:
            key, value = part.split("=")
            if key in ["scale", "offset", "min_step", "max_step"]:
                config[key] = float(value)
            elif key == "label":
                config[key] = value

        file_configs.append(config)

        return file_configs


if __name__ == "__main__":
    # file_configs = parse_cli_args()
    file_configs = [
        {
            "label": "SEAC (Penalty=0.005)",
            "file": "results/sacred/6/metrics.json",
            "scale": 20,
            "offset": 20e6,
        },
        {
            "label": "SEAC (Penalty=0.5)",
            "file": "results/sacred/7/metrics.json",
            "scale": 20,
            "offset": 30e6,
        },
    ]

    # Run the main function
    main(file_configs, "figures/tiny_1p_SEAC_PenaltyTest.png")
