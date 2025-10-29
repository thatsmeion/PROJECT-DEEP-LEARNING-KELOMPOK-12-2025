from roboflow import Roboflow
import os
import cv2
import matplotlib.pyplot as plt

rf = Roboflow(api_key="d5YdekFzWjue5QamOna6")
project = rf.workspace("project-deep-learning-kelompok-12").project("tomato-detection-fresh-or-rotten-using-yolov8-7jarn")
version = project.version(1)
dataset = version.download("yolov8")


image_dir = os.path.join(dataset.location, "train", "images")

print("Daftar file di train/images:", os.listdir(image_dir)[:5])  # tampilkan 5 file pertama

sample_image = os.path.join(image_dir, os.listdir(image_dir)[0])

img = cv2.imread(sample_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.title("Contoh Gambar Dataset Tomat")
plt.show()
