import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

# Paths to the input and output logs
input_log_path = "/home/smithr38/Thesis-Project/epymarl-main/results/tb_logs/tiny_0_005penalty_1p_seed192884176"  # Path to the original event log
output_log_path = "/home/smithr38/Thesis-Project/epymarl-main/results/tb_logs/tiny_1p_0-005penalty_20m-30m"  # Directory to save the filtered log
max_steps = 30.01e6  # Maximum step to include in the filtered log

# Ensure output directory exists
os.makedirs(output_log_path, exist_ok=True)

# Load the original log
event_acc = EventAccumulator(input_log_path)
event_acc.Reload()

# Create a new TensorBoard writer
writer = SummaryWriter(output_log_path)

# Filter scalar data and write to the new log
for tag in event_acc.Tags()["scalars"]:
    scalar_events = event_acc.Scalars(tag)  # Get all scalar events for this tag
    for event in scalar_events:
        if event.step <= max_steps:
            writer.add_scalar(tag, event.value, event.step)

# Close the writer
writer.close()

print(f"Filtered TensorBoard log saved to: {output_log_path}")
