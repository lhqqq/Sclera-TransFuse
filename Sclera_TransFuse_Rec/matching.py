import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
import time
import torchvision.models as models
from PIL import Image
from PIL import Image as PILImage

import yaml
from cross_efficient_vit import CrossEfficientViT
# Check if GPU is available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
opt = "/sdata/haiqing.li/seg/LightCNNresNEW/configs/architecture.yaml"
with open(opt, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

model_path =

checkpoint = torch.load(model_path)
model = CrossEfficientViT(config=config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
def clahe_hsv(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def to_clahe_hsv_twice(image):  # 输入是rgb
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    img = clahe_hsv(img)
    img = clahe_hsv(img)
    im = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return im

transform = transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.Lambda(to_clahe_hsv_twice),
                      transforms.ToTensor(),

                  ])

def load_img(img_path, device):
    image = PILImage.open(img_path).convert('RGB')
    c_img = transform(image)

    c_img = torch.unsqueeze(c_img, dim=0).to(device)
    return c_img


def extract_and_cache_features(image_path, feature_cache):
    if image_path in feature_cache:

        return feature_cache[image_path]
    else:

        with torch.no_grad():
            img_tensor = load_img(image_path, device)
            _,features = model(img_tensor)
        # Cache the extracted feature
        feature_cache[image_path] = features.squeeze().cpu().numpy()
        return feature_cache[image_path]

# Path to your dataset folder
dataset_folder =

# Get a list of image files in the dataset folder
image_files = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith('.jpg'):
            image_files.append(os.path.join(root, file))
            # print(image_files)


feature_cache = {}

num_samples = len(image_files)  # Initialize the sample counter
start=time.time()
cout=0
for image_path in image_files:
    # cout+=1
    extract_and_cache_features(image_path, feature_cache)
    # num_samples += 1  # Increment the sample counter for each processed image
end = time.time()
print(end-start)
print(cout)


same_class_scores = []
different_class_scores = []

# Compare features of each image with all other images in the dataset
for i, image1_path in enumerate(image_files):
    class1 = image1_path.split(os.path.sep)[-1].split('_')[0]  # Extract class label
    # print('1',class1)
    feature1 = extract_and_cache_features(image1_path, feature_cache)

    for j, image2_path in enumerate(image_files):
            class2 = image2_path.split(os.path.sep)[-1].split('_')[0]  # Extract class label
            # print('2',class2)
            feature2 = extract_and_cache_features(image2_path, feature_cache)

            # Calculate cosine similarity between the two features
            similarity_score = cosine_similarity([feature1], [feature2])[0][0]
            # print(similarity_score)

            # Determine if they belong to the same class
            if class1 == class2:
                same_class_scores.append(similarity_score)

            else:
                different_class_scores.append(similarity_score)

print('1',len(same_class_scores))
print('2', len(different_class_scores))


same_class_scores = np.array(same_class_scores)
different_class_scores = np.array(different_class_scores)

np.save('./1.npy', same_class_scores)
np.save('./2.npy', different_class_scores)
print("Total Samples:", num_samples)  # Print the total number of   samples
print("Same Class Scores:")
# print(same_class_scores)
print("Different Class Scores:")
print(different_class_scores)
