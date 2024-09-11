import os
import random
import shutil


source_dir = "C:\\Users\\anton\\MachineLearning\\PointsCollision\\datasets\\128_14_4_test_multiclass"
dest_dir = "C:\\Users\\anton\\MachineLearning\\PointsCollision\\datasets\\128_14_4_test_multiclass_s"

# Get list of image files
image_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

random.shuffle(image_files)

for i, image in enumerate(image_files):
    new_path = os.path.join(dest_dir, f"{i}_{image.split('_')[1]}")
    shutil.copy(os.path.join(source_dir, image), new_path)
