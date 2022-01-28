
# Window Detection on Jetson Nano 2GB Developer Kit using Yolov5.

Window detection system which will detect the quality of a window,
identify whether it’s clear, foggy, clean and unclean and then recommend the steps to take
after identifying the property of the window.
## Aim and Objectives

### Aim

To create a Window detection system which will detect the quality of a window,
identify whether it’s clear, foggy, clean and unclean and then recommend the steps to take
after identifying the property of the window.

### Objectives

• The main objective of the project is to create a program which can be either run on
Jetson nano or any pc with YOLOv5 installed and start detecting using the camera
module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine
learning.

• To show on the optical viewfinder of the camera module whether a window is clean
or unclean.
## Abstract

• A window’s cleanliness can be detected by the live feed derived from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning
(ML), where machines are trained to identify various objects from one another. Machine
Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.

• Clear and Clean windows generally provide with a good ambience for either a business
store or a residential home.

• Clean windows make the room a lot brighter when compared to unclean or foggy ones
and lets the necessary Sun rays to enter the room eluminating it.
## Introduction

• This project is based on a Window detection model with modifications. We are going to
implement this project with Machine Learning and this project can be even run on jetson
nano which we have done.

• This project can also be used to gather information about Window condition, i.e., Clean,
Unclean.

• Windows can be classified into clean, unclean, foggy, clear, dirty, spotless etc based on
the image annotation we give in roboflow.

• In a Window because of various textures and railings Sometimes it is difficult to detect
various spots on it for ex: - Dew drops or mild rain drops. The detection of these small
spots results in making the model better trained.

• Also, the texture on windows sometimes makes it difficult for the model to realize the
difference between clean and unclean window.

• Neural networks and machine learning have been used for these tasks and have obtained
good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Window detection as well.
## Literature Review

• Windows are among the first thing that people see in a building. Unfortunately, many
people usually overlook the importance of having clean windows. Whether is
commercial window cleaning or high-rise window cleaning, having your windows
regularly cleaned can make a big difference.

• For business owners, this can be the difference between closing a sale and coming up
empty-handed, since an office building with sparkling windows indicates to your client
that you are conscientious and willing to take care of even the smallest of details.

• The human psyche is wired to magnify faults, however negligible they may be, and to
general goodness especially where competing interests must be factored in. That is
what the difference in the quality of glass windows between two houses on sale does.

• Window cleaning helps to allow more natural light. With time glasses in windows
usually become dull due to contaminants such as oxidation, hard mineral, acid rain,
paint, spray among others that prevents natural light from entering inside the building.

• Dirt, dust and grime usually tend to settle on windows over time if they are not
regularly cleaned. They not only make the window to look dull, but it also increases
the growth of allergen that can cause allergic reaction, skin problem and respiratory
problems.
## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.

## Jetson Nano 2GB



https://user-images.githubusercontent.com/89011801/151482100-936d971d-0673-4ff0-b3a0-1093f4687b30.mp4





## Proposed System

1. Study basics of machine learning and image recognition.
    
2. Start with implementation
        
        ➢ Front-end development
        ➢ Back-end development

3. Testing, analysing and improvising the model. An application using python and Roboflow
and its machine learning libraries will be using machine learning to identify the clarity of
windows.

4. Use datasets to interpret the windows and suggest whether the windows are clear or
unclean.
## Methodology

The window detection system is a program that focuses on implementing real time window
detection.

It is a prototype of a new product that comprises of the main module:

Window detection and then showing on viewfinder whether clean or unclean.

Window Detection Module

```bash
This Module is divided into two parts:
```

    1] Window detection

• Ability to detect the location of window in any input image or frame. The output is the
bounding box coordinates on the detected window.

• For this task, initially the Dataset library Kaggle was considered. But integrating it
was a complex task so then we just downloaded the images from gettyimages.ae and
google images and made our own dataset.

• This Datasets identifies windows in a Bitmap graphic object and returns the bounding
box image with annotation of windows present in each image.

    2] Clarity Detection

• Classification of the window based on whether it is clean or unclean.

• Hence YOLOv5 which is a model library from roboflow for image classification and
vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in
production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments is
straightforward.

• YOLOv5 was used to train and test our model for various classes like clean, unclean.
We trained it for 149 epochs and achieved an accuracy of approximately 92%.
## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```
```bash
sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl
python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer
libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-
qt4 python-qt4-gl libgle3 python-dev libssl-dev libpq-dev python-dev libxml2-dev libxslt1-
dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev
```
```bash
vim ~/.bashrc
```
####################### add line ####################
```bash
export OPENBLAS_CORETYPE=ARMV8
```

```bash
source ~/.bashrc
```
```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```
## Window Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder
link of project


## Running Window Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo




https://user-images.githubusercontent.com/89011801/151315891-c3561fc7-0621-4c8f-9cbb-16096d34f4e5.mp4




## Advantages

• The Window detection system will be of great advantage where a user has lack of time,
motivation, unwell or differently abled.

• It will be useful to users who are very busy because of work or are because of prior
schedules.

• Just place the viewfinder showing the window on screen and it will detect it.

• It will be faster to just then clean windows using minimal or very less workforce.
## Application

• Detects Window clarity in each image frame or viewfinder using a camera module.

• Can be used to clean windows when used with proper hardware like machines which can
clean.

•
Can be used as a reference for other ai models based on window detection
## Future Scope


• As we know technology is marching towards automation, so this project is one of the step
towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

• Cleaning windows of vehicles as well as ships, aquarium windows can also be considered
as an future scope for our project.
## Conclusion

▪ In this project our model is trying to detect windows for whether they are clean or
unclean and then showing it on viewfinder live as what the state of window is.

▪ This model solves the basic need of having a clean and clear window for our users
who because of lack of time or other reasons are unable to keep their windows clean.

▪ It can even ease the work of people who are in the sanitization industry or the cleaning
industry and save them a lot of time and money.
## Refrences

#### 1]Roboflow :- https://roboflow.com/

#### 2] Datasets or images used: https://www.gettyimages.ae/photos/window?family=editorial&assettype=image&phrase=window&sort=mostpopular

#### 3] Google images
## Articles

#### [1] https://wedowindowssantafe.com/importance-of-window-cleaning/

#### [2] https://laborpanes.com/blog/7-reasons-you-should-get-a-residential-window-cleaning/
