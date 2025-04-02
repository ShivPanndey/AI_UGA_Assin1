# AI_UGA_TakeHomeAssignment: Edge Detection using U-Net and Classical Methods
# Edge Detection using U-Net and Classical Methods

This project performs edge detection on images from the BSDS500 dataset using both classical computer vision methods (Sobel, Canny) and a deep learning approach (U-Net). The goal is to compare traditional edge detection techniques with the results from a U-Net trained for edge segmentation.

## 📁 Dataset

- **Source**: [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **Path Used**: `/BSDS500/data/images/test/`

## 🛠️ Features

- Classical Edge Detection (Sobel X, Y, XY, and Canny)
- Deep Learning Edge Detection using U-Net
- Visualization of original images and predicted edge masks
- Sigmoid normalization and thresholding for binary mask generation


## 🔗 Main Project Notebook
 [Link to main project code](http://localhost:8888/lab/tree/AI_UGA/AIUGA_assin1.ipynb)


## 📦 Dependencies
Make sure you have the following packages installed:
```bash
pip install torch torchvision matplotlib opencv-python pillow







