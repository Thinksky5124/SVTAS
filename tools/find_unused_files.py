import os

# Define paths for the folder and the two text files
folder_path = "/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/groundTruth"
old_text_file_1 = "/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/splits/testLongOnly_test.bundle" ### test file
old_text_file_2 = "/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/splits/testLongOnly_train.bundle" ### train file 

# New text file names (to avoid overwriting the originals)
new_text_file_1 = "/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/splits/ext_testLongOnly_test.bundle"
new_text_file_2 = "/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/splits/ext_testLongOnly_train.bundle"

# Get the set of filenames in the folder
folder_files = set(os.listdir(folder_path))

# Read the contents of the old text files into sets and lists
with open(old_text_file_1, 'r') as f1:
    old_entries_1 = [line.strip() for line in f1 if line.strip()]
with open(old_text_file_2, 'r') as f2:
    old_entries_2 = [line.strip() for line in f2 if line.strip()]

# Create a set of already listed files from both text files
already_listed = set(old_entries_1) | set(old_entries_2)

# Identify files that are in the folder but not already listed
files_not_listed = folder_files - already_listed

# Prepare new entries based on the file's number of lines
new_entries_1 = []  # for files with > 1800 lines
new_entries_2 = []  # for files with <= 1800 lines

for filename in files_not_listed:
    file_path = os.path.join(folder_path, filename)
    
    # Only process if it's a file
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as file:
                num_lines = sum(1 for _ in file)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if num_lines > 1800:
            new_entries_1.append(filename)
        else:
            new_entries_2.append(filename)

# Combine old and new entries for each new text file
combined_entries_1 = old_entries_1 + new_entries_1
combined_entries_2 = old_entries_2 + new_entries_2

# Write the combined entries to new text files
with open(new_text_file_1, 'w') as nt1:
    for name in combined_entries_1:
        nt1.write(name + "\n")

with open(new_text_file_2, 'w') as nt2:
    for name in combined_entries_2:
        nt2.write(name + "\n")

print(f"Total entries in {new_text_file_1}: {len(combined_entries_1)}")
print(f"Total entries in {new_text_file_2}: {len(combined_entries_2)}")