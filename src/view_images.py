import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "Dataset/archive"

# Select one class
class_name = "normal_columnar"

class_path = os.path.join(dataset_path, class_name)

# Get image files
images = os.listdir(class_path)

print("Class:", class_name)
print("Number of images:", len(images))

# Select first image
image_name = images[0]
image_path = os.path.join(class_path, image_name)

# Load image
image = Image.open(image_path)

print("Image name:", image_name)
print("Image size:", image.size)
print("Image mode:", image.mode)

# Display image
plt.imshow(image)
plt.title(class_name)
plt.axis("off")
plt.show()