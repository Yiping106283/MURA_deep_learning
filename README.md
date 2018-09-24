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

* For localization task, we use class activation map. 
  We may further try GradCAM in the future. 
  Following is an example of the CAM.
  <figure class="half">
    <img src="https://github.com/GoAhead106283/MURA_deep_learning/blob/master/image2.png" title="original image" width="100" />
    <img src="https://github.com/GoAhead106283/MURA_deep_learning/blob/master/image2_heatmap.jpg" title="CAM" width="100" />
  </figure>
  
  
* For retrieval task, We use CBIR(Content Based Image Retrieval). The result of Image Retrieval is output as MURA-  v1.0/train/XR_ELBOW/patient04989/study1_positive/image1.png. 
  Here is an example of our result:
  <figure class="half">
    <img src="https://github.com/GoAhead106283/MURA_deep_learning/blob/master/query.png" title="original image" width="100" />
    <img src="https://github.com/GoAhead106283/MURA_deep_learning/blob/master/answer.png" title="CAM" width="100" />
  </figure>
 

* We provide two example pictures for localization and image retrieval, respectively. The example of heat map input is image2.png, the example of image retrieval input is image1.png. 

References
------
B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, andA. Torralba. Learning deep features for discriminativelocalization. InProceedings of the IEEE Conference onComputer Vision and Pattern Recognition, pages2921–2929, 2016.

H. Hebbar, S. Mushigeri, and U. Niranjan. Medicalimage retrieval–performance comparison using texturefeatures.International Journal of Engineering Researchand Development, 9(9):30–34, 2014
