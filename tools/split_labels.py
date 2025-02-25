import os
import re

def process_label_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all text files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)

            # Output file paths
            base_name = os.path.splitext(filename)[0]
            output_b_number_path = os.path.join(output_folder, f"{base_name}_branch.txt")
            output_word_path = os.path.join(output_folder, f"{base_name}.txt")
          
            with open(input_path, "r") as infile, \
                 open(output_b_number_path, "w") as b_number_file, \
                 open(output_word_path, "w") as word_file:
                
                for line in infile:
                    # Split the line into index and label
                    parts = line.strip().split(": ", 1)
                    if len(parts) == 2:
                        label = parts[1]

                        if label == "background":
                            # Write "background" to both files
                            b_number_file.write("background\n")
                            word_file.write("background\n")
                        else:
                            # Match the pattern for "bNUMBERWORD"
                            match = re.match(r'(b\d+)([a-zA-Z]+)', label)
                            if match:
                                b_number = match.group(1)
                                word = match.group(2)
                                # Write to respective output files
                                b_number_file.write(f"{b_number}\n")
                                word_file.write(f"{word}\n")

    print(f"Processing complete. Output files saved in {output_folder}")

# Specify the input and output folder paths
input_folder = "data/thal_new/groundTruth"
output_folder = "data/thal_new/groundTruth_split"

# Process the files
process_label_files(input_folder, output_folder)
