import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset
dataset_path = "Dataset/archive"

# New processed dataset
output_path = "Dataset/processed"

# Normal classes
normal_classes = [
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel"
]

# Abnormal classes
abnormal_classes = [
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ"
]

images = []
labels = []


# -----------------------------
# Collect NORMAL images
# -----------------------------

for class_name in normal_classes:

    class_path = os.path.join(dataset_path, class_name)

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        # Use only original .BMP images
        # Ignore segmentation masks ending with -d.bmp
        if (
            os.path.isfile(file_path)
            and filename.lower().endswith(".bmp")
            and not filename.lower().endswith("-d.bmp")
        ):
            images.append(file_path)
            labels.append("normal")


# -----------------------------
# Collect ABNORMAL images
# -----------------------------

for class_name in abnormal_classes:

    class_path = os.path.join(dataset_path, class_name)

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        # Use only original .BMP images
        # Ignore segmentation masks ending with -d.bmp
        if (
            os.path.isfile(file_path)
            and filename.lower().endswith(".bmp")
            and not filename.lower().endswith("-d.bmp")
        ):
            images.append(file_path)
            labels.append("abnormal")


# -----------------------------
# Check collected data
# -----------------------------

print("Total actual cell images:", len(images))
print("Normal images:", labels.count("normal"))
print("Abnormal images:", labels.count("abnormal"))


# -----------------------------
# Train / Validation / Test
# -----------------------------

# 70% training, 30% temporary
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Split temporary 50/50
# → 15% validation
# → 15% test
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)


# -----------------------------
# Create folders
# -----------------------------

splits = {
    "train": (train_images, train_labels),
    "validation": (val_images, val_labels),
    "test": (test_images, test_labels)
}

for split_name, (split_images, split_labels) in splits.items():

    for category in ["normal", "abnormal"]:

        folder = os.path.join(
            output_path,
            split_name,
            category
        )

        os.makedirs(folder, exist_ok=True)

    # Copy images
    for image_path, label in zip(split_images, split_labels):

        filename = os.path.basename(image_path)

        destination = os.path.join(
            output_path,
            split_name,
            label,
            filename
        )

        shutil.copy2(image_path, destination)


# -----------------------------
# Final result
# -----------------------------

print()
print("Dataset split completed!")
print()

print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Testing images:", len(test_images))
print("Total images:", len(images))