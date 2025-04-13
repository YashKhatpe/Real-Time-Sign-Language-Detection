import json

# Load the list of 100 words
with open("words.txt", "r") as f:
    selected_words = {line.strip().lower() for line in f}  # Convert to lowercase for consistency

# Load the full WLASL dataset
with open("WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

# Dictionary to store video URLs
video_links = {}

for entry in wlasl_data:
    gloss = entry["gloss"].lower()  # Convert to lowercase for matching
    if gloss in selected_words:
        for instance in entry["instances"]:
            video_id = str(instance["video_id"])
            video_links[video_id] = instance["url"]

# Save extracted links
with open("healthcare_100_videos.json", "w") as f:
    json.dump(video_links, f, indent=4)

print(f"Extracted {len(video_links)} video links for download.")
