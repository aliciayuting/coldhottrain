import json

# Load the JSON file
with open("/share/desa/nfs02/cold/test/preselect_grads.json", "r") as f:
    data = json.load(f)

# List all top-level keys
print(data['percent'])
print(list(data['layers'].keys()))