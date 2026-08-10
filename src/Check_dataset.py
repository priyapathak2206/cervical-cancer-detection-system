import os

dataset_path = "Dataset/archive"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)
        print(folder, ":", len(images), "images")