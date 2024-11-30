import json

from torch.utils.tensorboard import SummaryWriter

json_file_path = "/home/smithr38/Thesis-Project/results/sacred/5/metrics.json"
log_dir = f"/home/smithr38/Thesis-Project/results/tb_logs/small_1p"
offset = 40e6

# Load JSON data
with open(json_file_path, "r") as f:
    data = json.load(f)

# Extract the relevant data
steps = data["episode_reward"]["steps"]
values = data["episode_reward"]["values"]

# Ensure steps and values are of the same length
if len(steps) != len(values):
    raise ValueError("Mismatch between the number of steps and values in the JSON data.")

# Create a TensorBoard log directory
writer = SummaryWriter(log_dir)

# Write the data to TensorBoard log
for step, value in zip(steps, values):
    writer.add_scalar("return_mean", value, (step * 20) + offset)

# Close the writer
writer.close()

print(f"TensorBoard logs created in: {log_dir}")
