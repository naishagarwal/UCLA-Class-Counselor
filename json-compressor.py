#compressing json files into one singular json file
import json

json_files = ['prompts_classes.json', 'prompts_major_department.json', 'prompts_major_school.json', 'random_prompts.json']

combined_data = []

# Iterate over each JSON file in the list
for json_file in json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
        combined_data.extend(data)

# Write the combined data to the output file
output_file = 'final_prompts.json'
with open(output_file, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)