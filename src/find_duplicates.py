import os
import hashlib
from collections import defaultdict

dataset_path = "Dataset/archive"

# Store hash -> list of files
file_hashes = defaultdict(list)

print("Scanning images...")

for root, folders, files in os.walk(dataset_path):

    for filename in files:

        if filename.lower().endswith(".bmp"):

            file_path = os.path.join(root, filename)

            # Read image and calculate hash
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            file_hashes[file_hash].append(file_path)


# Find duplicate groups
duplicate_groups = []

for file_hash, files in file_hashes.items():

    if len(files) > 1:
        duplicate_groups.append(files)


print("\nRESULT")
print("-----------------------------")

print("Total unique images:", len(file_hashes))

print("Duplicate groups:", len(duplicate_groups))

duplicate_count = sum(len(group) - 1 for group in duplicate_groups)

print("Duplicate files:", duplicate_count)


print("\nExample duplicate groups:")
print("-----------------------------")

for group in duplicate_groups[:5]:

    print("\nDuplicate group:")

    for file in group:
        print(" ", file)