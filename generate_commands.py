import os
import json

# Create output directory
output_dir = "commands_dataset"
os.makedirs(output_dir, exist_ok=True)

wire_colors = ["red", "green", "blue"]


i_data = 0
for wire in wire_colors: 
    for terminal in range(0,9):
        data_point = {
            "prediction_action": ["lock"],
            "predicted_wire": wire,
            "predicted_terminal": f"terminal_{terminal}",
        }

        output_path = os.path.join(output_dir, f"predicted_wires_dataset_{i_data}.json")
        with open(output_path, "w") as f:
            json.dump(data_point, f, indent=2)
            print(f"Dataset {i_data} successfully created: {output_path}")

        i_data += 1

