import json

# Load the full gloss dataset (WLASL_v0.3.json)
with open("WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

# List of 100 healthcare-related words (you can refine this)
healthcare_glosses = [
    "who", "what", "where", "when", "why", "how", "yes", "no", "please", "thank you", 
    "help", "stop", "go", "wait", "open", "close", "more", "less", "fast", "slow", 
    "doctor", "nurse", "patient", "hospital", 
    "emergency",
    "receptionist", "appointment", "sick", "pain", 
    "cold", "cough", 
    "headache", "stomachache", "dizzy", 
    "allergy", 
    "vomit", "weak", "tired", "stress", "infection", 
    "test", "blood", "pressure",
    "medicine", 
    "rest", "breathe", "drink", "eat", "sleep", "exercise", "clean", 
    "wash", "sit", "stand", "lie down", "move", "fire", "police", "accident", 
    "danger", 
    "choke", "heart attack", 
    "safe",
    "water", "food", "bathroom", 
    "phone", "family", "friend", "home", "insurance", 
    "report", "sign", 
    "write", "read", "understand", "repeat"
    # "pharmacy", 
    # "ambulance", 
    # "fever",
    # "nausea",
    # "injury", "burn", "bleeding",
    # "swelling", "rash", "check-up",
    # "injection", "capsule", "tablet", 
    # "warning", "poison",
    # "stroke", "faint", "broken", "lost", 
    #   "unsafe", "evacuate", 
    # "id"
]


# Create a mapping of gloss words to their corresponding instances
gloss_instances_map = {
    entry["gloss"]: entry["instances"] for entry in wlasl_data if entry["gloss"] in healthcare_glosses
}

# Load the full dataset with video mappings (nslt_2000.json)
with open("nslt_2000.json", "r") as f:
    nslt_data = json.load(f)

# Extract only the relevant video samples
filtered_data = {
    vid: data for vid, data in nslt_data.items() if any(
        instance["video_id"] == vid for gloss, instances in gloss_instances_map.items() for instance in instances
    )
}

# Save the filtered dataset
with open("healthcare_100.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print(f"Extracted {len(filtered_data)} video samples for the healthcare domain.")