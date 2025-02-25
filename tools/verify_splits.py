# Read the filenames from both files into sets
with open('data/THAL_CLSWISE/thal/splits/test.split1.bundle', 'r') as f1, open('data/THAL_CLSWISE/thal/splits/train.split1.bundle', 'r') as f2:
    set1 = set(f1.read().splitlines())
    set2 = set(f2.read().splitlines())

# Find the common files
print(set1)
print('\n')
print(set2)
common_files = set1.intersection(set2)

if common_files:
    print("Common files:", common_files)
else:
    print("No common files.")
