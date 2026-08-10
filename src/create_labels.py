import os

dataset_path = "Dataset/archive"

normal_classes = [
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel"
]

abnormal_classes = [
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ"
]

normal_count = 0
abnormal_count = 0

print("NORMAL IMAGES")
print("----------------")

for class_name in normal_classes:
    class_path = os.path.join(dataset_path, class_name)
    count = len(os.listdir(class_path))
    normal_count += count
    print(class_name, ":", count)

print("\nABNORMAL IMAGES")
print("----------------")

for class_name in abnormal_classes:
    class_path = os.path.join(dataset_path, class_name)
    count = len(os.listdir(class_path))
    abnormal_count += count
    print(class_name, ":", count)

print("\nTOTAL")
print("----------------")
print("Normal:", normal_count)
print("Abnormal:", abnormal_count)
print("Total:", normal_count + abnormal_count)