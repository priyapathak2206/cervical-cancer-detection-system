from PIL import Image
import matplotlib.pyplot as plt

image1_path = "Dataset/archive/normal_columnar/153956040-153956058-001-d.bmp"
image2_path = "Dataset/archive/normal_columnar/153956040-153956058-001.BMP"

image1 = Image.open(image1_path).convert("RGB")
image2 = Image.open(image2_path).convert("RGB")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title("001-d.bmp")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title("001.BMP")
plt.axis("off")

plt.tight_layout()
plt.show()