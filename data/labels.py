import json

# Load the dataset
with open("healthcare_100.json", "r") as f:
    data = json.load(f)

# Extract action IDs and assign labels
action_to_word = {entry["action"][0]: f"word_{entry['action'][0]}" for entry in data.values()}

# Save the mapping
with open("labels.json", "w") as f:
    json.dump(action_to_word, f, indent=4)
