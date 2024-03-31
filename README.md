# Enhance Construction Site Safety by Detecting PPE with YoloV8

Final Project, MIT 15.773 - Hands-on Deep Learning
Team Members: Nuobei Zhang, Xidan Xu, [Ikechukwu Ume](https://www.linkedin.com/in/ikechukwu-ume-p-e-90217350/), [Safiyah Gold](https://www.linkedin.com/in/safiyahgold/)

# Problem and Motivation

Each year, there are $3-7k ​OSHA non-compliance fines per construction site related incident in the United States, and cost ​$5 billion in healthcare, lost income & production​. High risk of accidents and injuries ​caused by non-compliance with Personal Protective Equipment​(PPE). 

With the advances in the field of deep learning, we can apply Object Detection for PPE detection​, thus improve workforce safety and reduce non-compliance costs​.

# Task Details
## Data Source
We used the "Construction Site Safety Image Dataset" available on Roboflow via Kaggle. This dataset includes over 4,000 images, each meticulously labeled to showcase instances of both compliance and non-compliance with PPE protocols across varied construction settings. [Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

## Data Format

# Models and Result
We tried out various method for this object detection task. 

1. Pre-trained Models 
We started by testing the pre-trained YOLOv8 model on our unique dataset. This initial approach performed poorly because the model was pre-trained on the COCO dataset where the type of data (non-human highly varied objects) differs significantly from our target objects (humans, masks, helmets, and saftey vests). 

Left: Object detected using pre-trained model directly out of box with our data. Right: Ground Truth. This shows what we expected the model to detect. Notice that the accuracy boxes have the right annotations.	 
 
2. Transfer Learning  
We then took the backbone of the YOLOv8 model, pre-trained on COCO, and ran a Feature Pyramid Network (FPN) layer to adapt the model to our dataset. Here we froze the backbone layers (which extract the general features from the images) and only trained the head of the model. This resulted in a total of 1,884,528 trainable paramaters. We observed that the model did a relatively good job given how much our images looked to the COCO dataset. On reflection, we knew there was room for improvement so we set the patience equal to 3 so that it would early stop if model performance did not improve enough within three epochs. Additionally, the transfer learning model stopped at twenty-first epoch.  
 
Left: Object detected using transfer learning method. Right: Ground Truth. This example shows that the transfer learning method could adequately identify missing PPE (masks, hardhats, and safety vests). 
  
 
3. Fine Tuning 
We fine tuned the pre-trained YOLOv8 model on COCO and ran a Feature Pyramid Network (FPN) layer to better adapt the model to our dataset. In this step, we allowed all the parameters, including the parameters on the backbones, to be trainable. This resuted in a total of 3,423,150 trainable paramaters and 11,168 non-trainable paramaters. Our improved model with early-stopping and patience equal to 3, stopped seeing improvements after the nineteenth epoch. 
 
Left: Object detected using fine tuning on the pretrained model. Right: Ground Truth. This example shows that the transfer learning method shows significant improvement over transfer learning in this case. 
 
Left: Object detected using fine tuning on the pretrained model. Right: Ground Truth. This example shows that fine tuning using FPN delivered superior results over transfer learning alone. 
