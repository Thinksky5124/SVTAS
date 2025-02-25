import os
import re

def process_mapping_file(input_file, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Output file path
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file_path = os.path.join(output_folder, f"{base_name}_unique_labels.txt")

    # Sets to store unique labels
    unique_b_numbers = set()
    unique_words = set()

    with open(input_file, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            # Split the line into index and label
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                label = parts[1]

                if label == "background":
                    # Ensure "background" is only written once
                    if "background" not in unique_b_numbers:
                        outfile.write("background\n")
                        unique_b_numbers.add("background")
                else:
                    # Match the pattern for "bNUMBERWORD"
                    match = re.match(r'(b\d+)([a-zA-Z]+)', label)
                    if match:
                        b_number = match.group(1)
                        word = match.group(2)
                        # Write unique bNUMBER and word
                        if b_number not in unique_b_numbers:
                            outfile.write(f"{b_number}\n")
                            unique_b_numbers.add(b_number)
                        if word not in unique_words:
                            outfile.write(f"{word}\n")
                            unique_words.add(word)

    print(f"Processing complete. Output file saved at {output_file_path}")

# Specify the input file and output folder paths
input_file = "data/thal/mapping.txt"
output_folder = "data/thal/mapping_split.txt"

# Process the mapping file
process_mapping_file(input_file, output_folder)
