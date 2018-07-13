MURA Deep Learning Project --- Implementation in PyTorch
======
Prerequisites
------
* CUDA Toolkit v9.0
* cuDNN v7.1.2
* PyTorch v0.4.0
* OpenCV v3.4.1


Usage
------
* Before run Main.py, please run prepocessData.sh first
* Run Main.py, to get validation result, CAM heatmap, ImageRetrieval

Our Methods and Results
------
* For classification task, we use multi-instance learning. Since the task is a binary classification, we replaced the last fully connected layer of densenet201 with one that has a single ouput. Then we placed a sigmoid function after the output. For the loss function, we use BCE loss.

* For localization task, we use class activation map, see at https://arxiv.org/abs/1512.04150 . 
  We may further try GradCAM in the future.
  
* For retrieval task, We use CBIR(Content Based Image Retrieval). The result of Image Retrieval is output as MURA-v1.0/train/XR_ELBOW/patient04989/study1_positive/image1.png

* We provide two example pictures for localization and image retrieval, respectively. The example of heat map input is image2.png, the example of image retrieval input is image1.png. 
