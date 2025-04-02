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
🔍 Classical Edge Detection

The classical edge detection is done via OpenCV:
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

sobelx = cv2.Sobel(img_blur, cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(img_blur, cv2.CV_64F, dx=0, dy=1, ksize=5)
edges = cv2.Canny(img_blur, 100, 200)

🧠 Deep Learning Inference (U-Net)

Loads images using a PyTorch DataLoader (test_loader)
Performs inference on a pre-trained U-Net model
Applies sigmoid activation and thresholds to generate binary masks
Visualizes predicted edges alongside original images
with torch.no_grad():
    for images, img_paths in test_loader:
        outputs = model(images.to(device))
        predicted_mask = torch.sigmoid(outputs) > 0.5
        ...

📊 Visualization

Side-by-side comparison of:
Original Image
Predicted Edge Mask (U-Net)
Display handled with matplotlib

📝 Notes

Ensure your model is loaded properly and matches the image dimensions
Raw logits from the model are passed through sigmoid before thresholding
Squeeze is used to remove unnecessary dimensions from predictions
🤝 Contributing

Pull requests and suggestions are welcome!

## 📦 Dependencies
Make sure you have the following packages installed:
```bash
pip install torch torchvision matplotlib opencv-python pillow

Also ensure you are running this in a PyTorch-enabled environment (e.g., pytorch-env in JupyterLab).






