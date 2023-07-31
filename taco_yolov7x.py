# -*- coding: utf-8 -*-
"""TACO_YOLOv7x.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F6x8iY2AzC1yk3ctXAUriAcuc4I-TiMJ

# TACO (Trash Annotations in Context): Leveraging YOLOv7 for Precise Trash Detection and Classification

<br>

<img src="https://raw.githubusercontent.com/doguilmak/TACO-Trash-Annotations-in-Context-YOLOv7x/main/asset/contamination-4286704_1280.jpg" height=400 width=1000 alt="Contamination"/>


<small>Picture Source: <a href="https://pixabay.com/users/yogendras31-12827898/">Pixabay - yogendras31</a>

<br>

## Introduction

In the face of growing environmental challenges, it has become imperative to develop effective solutions for preserving our planet. One crucial aspect is the proper management of waste and the need to accurately identify and categorize trash objects in real-world contexts. The emergence of advanced computer vision techniques, such as the YOLO (You Only Look Once) algorithm, combined with high-quality annotated datasets like TACO (Trash Annotations in Context), holds tremendous potential for addressing these environmental concerns. This article highlights the significance of TACO data for leveraging YOLO in waste detection and management.

<br>

1. **Enabling Accurate Trash Object Detection**:
The TACO dataset plays a crucial role in training and validating YOLO models for detecting various types of trash objects in diverse environmental settings. By providing precise annotations of trash items in contextual images, TACO enhances the algorithm's ability to identify and classify different waste materials accurately. This accuracy is vital for the development of robust waste detection systems, enabling more efficient waste management practices.

2. **Improving Waste Sorting and Recycling Efforts**:
Annotated datasets like TACO are invaluable in enhancing waste sorting and recycling efforts. By utilizing YOLO models trained on TACO data, it becomes possible to automatically identify and sort different types of trash, including plastics, paper, glass, metals, and organic waste. This capability not only streamlines waste management processes but also promotes recycling initiatives, reducing the amount of waste that ends up in landfills or polluting our environment.

3. **Supporting Smart Waste Management Systems**:
The combination of TACO data and YOLO-based object detection algorithms has the potential to revolutionize waste management systems. By deploying cameras or drones equipped with YOLO models trained on TACO, it becomes feasible to monitor waste accumulation in real-time, detect illegal dumping, and optimize waste collection routes. This proactive approach enables timely intervention and more efficient allocation of resources, leading to significant cost savings and a cleaner environment.

4. **Facilitating Environmental Monitoring and Research**:
TACO data provides a valuable resource for environmental monitoring and research initiatives. By accurately identifying and tracking trash objects in various habitats, researchers can gain insights into patterns of waste accumulation, identify pollution hotspots, and evaluate the effectiveness of waste management policies. Such data-driven analysis helps in designing targeted interventions and raising awareness about the impact of waste on ecosystems.

5. **Encouraging Collaboration and Innovation**:
The availability of high-quality annotated datasets like TACO encourages collaboration between researchers, data scientists, and environmental organizations. These datasets can be utilized to develop and benchmark novel approaches to waste detection, classification, and management. The collective efforts of the research community can lead to the development of more sophisticated algorithms, improving the accuracy and efficiency of waste-related applications.

<br>

## Dataset

you can easily access the TACO dataset in YOLO format by visiting the provided Kaggle link: [TACO Dataset in YOLO Format](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format). Once on the Kaggle page, you will be able to find more information about the TACO dataset, including its contents, annotations, and how it is formatted for use with YOLO-based models.

<br>

## Keywords

* Waste
* Recycling
* Pollution
* Trash bins
* YOLOv7 (Object detection algorithm)
* Object detection
* Deep learning

<br>

Make sure your runtime is **GPU** (_not_ CPU or TPU). And if it is an option, make sure you are using _Python 3_. You can select these settings by going to `Runtime -> Change runtime type -> Select the above mentioned settings and then press SAVE`.

## Importing Libraries
"""

from google.colab import files
from google.colab import drive

import os
import random
import shutil

import moviepy.editor as mp
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

"""## Data Preprocessing"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

files.upload()

!unzip -q /content/archive.zip

# !rm -rf /content/archive.zip

#@markdown ---
#@markdown ### Enter image paths:
train_images_dir = "/content/train/images" #@param {type:"string"}
val_images_dir = "/content/valid/images" #@param {type:"string"}

train_image_count = len([f for f in os.listdir(train_images_dir) if f.endswith(".jpg")])
val_image_count = len([f for f in os.listdir(val_images_dir) if f.endswith(".jpg")])

print(f"Number of images in train folder: {train_image_count}")
print(f"Number of images in val folder: {val_image_count}")
#@markdown ---

#@markdown ---
#@markdown ### Enter label paths:
train_labels_dir = "/content/train/labels" #@param {type:"string"}
val_labels_dir = "/content/valid/labels" #@param {type:"string"}

train_txt_count = len([f for f in os.listdir(train_labels_dir) if f.endswith(".txt")])
val_txt_count = len([f for f in os.listdir(val_labels_dir) if f.endswith(".txt")])

print(f"Number of TXT files in train labels folder: {train_txt_count}")
print(f"Number of TXT files in val labels folder: {val_txt_count}")
#@markdown ---

#@markdown ---
#@markdown You can plot any image and labels you want.

image_path = 'train/images/000087_jpg.rf.545747a3750e607ba2f81e4b205637a5.jpg' #@param {type:"string"}
label_path = 'train/labels/000087_jpg.rf.545747a3750e607ba2f81e4b205637a5.txt' #@param {type:"string"}

image = Image.open(image_path)
image_width, image_height = image.size
fig, ax = plt.subplots(1)
ax.imshow(image)

unique_classes = {}

with open(label_path, 'r') as file:
    for line in file:

        class_label, x, y, width, height = line.split()

        x = float(x) * image_width
        y = float(y) * image_height
        width = float(width) * image_width
        height = float(height) * image_height

        x_min = x - width / 2
        y_min = y - height / 2

        if class_label not in unique_classes:
            unique_classes[class_label] = f"C{len(unique_classes) + 1}"

        box_color = unique_classes[class_label]

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)


ax.set_aspect('equal')
legend_handles = [patches.Patch(color=color, label=f"Class {label}") for label, color in unique_classes.items()]
ax.legend(handles=legend_handles)
plt.show()
#@markdown ---

"""## Clone YOLO v7 and Train the Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

"""Following URL of the GitHub repository that contains the YOLOv7 implementation by WongKinYiu. The .git extension indicates that it is a Git repository.

When this command is executed, it will create a local copy of the entire repository on your machine. This allows you to access and use the YOLOv7 implementation provided by WongKinYiu for your own projects. You can then navigate to the cloned repository and access the relevant files and directories, including the training script, configuration files, and pre-trained weights.
"""

!git clone https://github.com/WongKinYiu/yolov7.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov7

"""You can change model from [WongKinYiu GitHub](https://github.com/WongKinYiu/yolov7) page. When this command is executed, it downloads the "yolov7x.pt" file from the specified URL. This pre-trained weights file can then be used as a starting point for training or as a pre-trained model for inference in YOLOv7-based projects.

Regarding other ".pt" files for YOLO models, there are different variations of the YOLO architecture, each with its own pre-trained weights. Some commonly used pre-trained weights files for YOLO models include:

* `yolov3.pt`: Pre-trained weights for YOLOv3.
* `yolov4.pt`: Pre-trained weights for YOLOv4.
* `yolov5s.pt`, yolov5m.pt, yolov5l.pt, yolov5x.pt: Pre-trained weights for YOLOv5, which is a different version developed by Ultralytics.

These pre-trained weights files can be obtained from various sources, including official repositories, GitHub releases, or community contributions. It's important to note that the architecture and compatibility of the weights file should match the specific YOLO version you are using in your project to ensure proper functionality.
"""

!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

"""Before training, you neet to go to `yolov7/data/coco.yaml` and define your number of class, class names and train-val paths like that:

```
# COCO 2017 dataset http://cocodataset.org

train: ../train/images
val: ../valid/images
test: ../test/images

nc: 18
names: ['Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton', 'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw', 'Styrofoam piece', 'Unlabeled litter']

roboflow:
  workspace: divya-lzcld
  project: taco-mqclx
  version: 3
  license: CC BY 4.0
  url: https://universe.roboflow.com/divya-lzcld/taco-mqclx/dataset/3
```

*   `!python train.py`: This is the command to execute the Python script *train.py* for training the YOLOv7 model.

*  `--device 0`: This parameter specifies the device (GPU) to be used for training. In this case, it is set to device 0, indicating the first GPU device.

*  `--batch-size 16`: This parameter determines the number of images in each batch during training. A batch size of *16* means that the model will process 16 images at a time before updating the weights.

*  `--data data/coco.yaml`: This parameter specifies the path to the YAML file containing the dataset configuration. In this case, **the *coco.yaml* file is used, which provides information about the dataset, including the classes and paths to the training and validation data.**

*  `--img 640 640`: This parameter sets the input image size for the model. The YOLOv7 model requires square input images, and here the dimensions are set to 640x640 pixels.

*  `--epochs 32`: This parameter defines the number of epochs, which represents the number of times the entire training dataset will be passed through the model during training. In this case, the model will be trained for **32 epochs**.

*  `--weights yolov7x.pt`: This parameter specifies the initial weights of the model. The *yolov7x.pt* file contains the pre-trained weights for the *YOLOv7* model, which will be used as the starting point for training.

*  `--hyp data/hyp.scratch.p5.yaml`: This parameter indicates the path to the YAML file containing hyperparameters for training. Hyperparameters include learning rate, weight decay, and other settings that affect the training process. Here, the *hyp.scratch.p5.yaml* file is used.

*  `--name yolov7x`: This parameter sets the name of the model during training. The name can be customized, and in this case, it is set to *yolov7x*.

If you are using GPU, try this:
"""

!python train.py --device 0 --batch-size 16 --data data/coco.yaml --img 640 640 --epochs 16 --weights yolov7x.pt --hyp data/hyp.scratch.p5.yaml --name yolov7x

"""If you are using only CPU, try this:"""

# !python train.py --device cpu --batch-size 16 --data data/coco.yaml --img 640 640 --epochs 64 --weights yolov7x.pt --hyp data/hyp.scratch.p5.yaml --name yolov7x

image = mpimg.imread('/content/yolov7/runs/train/yolov7x2/results.png')
plt.figure(figsize=(16, 9), dpi=300)
plt.title('Results')
plt.imshow(image)
plt.axis('off')
plt.show()

f1_curve = mpimg.imread('/content/yolov7/runs/train/yolov7x2/F1_curve.png')
pr_curve = mpimg.imread('/content/yolov7/runs/train/yolov7x2/PR_curve.png')

fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
axes[0].imshow(f1_curve)
axes[0].set_title('F1 Curve')
axes[0].axis('off')
axes[1].imshow(pr_curve)
axes[1].set_title('PR Curve')
axes[1].axis('off')
plt.show()

"""## Upload Test Image and Video to Make Prediction

Video Testing:

*   Select a video file that contains trash image footage.
*   Load the pre-trained YOLOv7 model and its corresponding weights.
*   Utilize the OpenCV library to read and process each frame of the video.
*   Pass each frame through the drone detection model for real-time inference.
*   Draw bounding boxes around detected buildings and display the annotated video output.


Image Testing:

*   Choose an image that includes a trash image.
*   Load the pre-trained YOLOv7 model and its weights.
*   Read and process the image.
*   Apply the trained model to the image and identify the presence of trash.
*   Visualize the image with a bounding box around the detected trash, if present.
*   Analyze the model's performance by assessing the correct identification of the trash classes.
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/
!mkdir test_data
# %cd /content/test_data

uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov7

"""Upload your images into test folder.

Image artist: [Skitterphoto](https://pixabay.com/users/skitterphoto-324082/)
"""

!python detect.py --weights runs/train/yolov7x2/weights/best.pt --conf 0.50 --img-size 640 --source /content/test_data/can-g55e33eb58_1280.jpg

#@markdown ### Plot detection
image = mpimg.imread('/content/yolov7/runs/detect/exp2/can-g55e33eb58_1280.jpg')
plt.figure(figsize=(16, 9), dpi=300)
plt.title('Results')
plt.imshow(image)
plt.axis('off')
plt.show()

"""## Contact Me
<p>If you have something to say to me please contact me:</p>

<ul>
  <li>Twitter: <a href="https://twitter.com/Doguilmak">Doguilmak</a></li>
  <li>Mail address: doguilmak@gmail.com</li>
</ul>
"""

from datetime import datetime
print(f"Changes have been made to the project on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")