
<!-- # Emoface -->
<h1 align="center">Exploring Generalized Face Encoding and Digital Biomarkers Using Visual Emotion Signals for Diagnosing Disorders in Adolescents</h1>
<p align="center">

## Installation 

### Dependencies

1) Install [conda](https://docs.conda.io/en/latest/miniconda.html)

<!-- 2) Install [mamba](https://github.com/mamba-org/mamba) -->

<!-- 0) Clone the repo with submodules:  -->
<!-- ``` -->
<!-- git clone --recurse-submodules ... -->
<!-- ``` -->

2) The relevant package versions are provided in the file `requirements38.txt`.








### Normal version

1) Pull the relevant submodules using: 
```bash
bash pull_submodules.sh
```


2) Set up a conda environment with one of the provided conda files. I recommend using `conda-environment_py38_cu11_ubuntu.yml`.  
<!-- This is the one I use for the cluster `conda-environment_py36_cu11_cluster.yml`. The differences between tehse two are probably not important but I include both for completeness.  -->


```bash
conda env create python=3.8 --file conda-environment_py38_cu11_ubuntu.yml
```
```bash
pip install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```


2) Activate the environment: 
```bash 
conda activate work38_cu11
```

3) For some reason cython is glitching in the requirements file so install it separately: 
```bash 
pip install Cython==0.29.14
```

4) Install `gdl` using pip install. I recommend using the `-e` option and I have not tested otherwise. 

```bash
pip install -e .
```

5) Verify that previous step correctly installed Pytorch3D


```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
```


## Usage 

Activate the environment: 
```bash
conda activate work38_cu11
```


## Structure 
1) [Data_processing](../Data_processing)



During the data preprocessing stage, the video frames are first extracted by running the script `video2image.py`. 


Next, `face_extraction.py` is executed to extract faces from the frames, removing background interference. 


Finally, `face_detection.py` is run to filter the data, discarding frames where no face is detected.




2) [Diagnosis_of_disease](../diagnosis_of_disease)



The auxiliary diagnosis stage is carried out by running the script `diagnosis_of_disease.py`. 
During this stage, the model simultaneously reads RGB images and pre-labeled files. 
The data is pre-split into training and validation sets in a 7:2 ratio. 


The training is performed using the weight file [epoch70.pth](Diagnosis_of_disease/epoch70.pth).



The backbone is extracted from torch.vision
Part of backbone is inserted to the original code, which would cause certain problem.  

Only resnet18:
The original code only support resnet18.  In this code, Resnet18 is added to backbones.




3) [Digital_target_statistics](Digital_target_statistics)

This portion of the data enters the digital target analysis stage:

The script `Gradient_computation.py` computes the gradient values of facial key points.

`Mathematical_Statistics.py` performs mathematical statistics to assess the significance of each key point across the entire visualized face.

Additionally, `Normalization.py` normalizes the numerical data and ranks the key points based on their significance.


4) [Facial_keypoint_detection](../Facial_keypoint_detection)

The `shape_predictor_68_face_landmarks.dat` file is a pre-trained model used for facial landmark detection. 
It provides 68 specific landmark points on a face, which correspond to various facial features such as the eyes, eyebrows, nose, mouth, and jawline. 
These landmarks are commonly used in tasks such as face alignment, facial expression analysis, and face recognition, enabling precise localization of facial features within an image.

The `Facial_keypoint_detection.py` script performs several critical steps for facial landmark detection. 
Initially, it loads the pre-trained model from the `shape_predictor_68_face_landmarks.dat` file, which is designed to identify 68 specific facial landmarks. 
The script then reads the input image and applies face detection algorithms to identify facial regions. Within these regions, the model is used to accurately locate the 68 facial key points. The detected landmarks are subsequently visualized on the image for inspection or further analysis. 

5) [Heatmap](../heatmap)

The `heatmap.py` code generates an attention heatmap based on a Gaussian distribution and overlays it on the input image, which is then saved as an image file. 
The central region of the attention heatmap is highlighted with brighter colors, indicating higher attention values in that area.

The `visualize_the_heat_map_statistics.py` code extracts a specified color range from the input image to create a new image. 
This new image has a white background and retains only the regions that fall within the specified color range. The resulting image is saved to a file and displayed using matplotlib.


6) [Standardized_digital_face](Standardized_digital_face)

The file `Rough_reconstruction.py` implements the rough reconstruction of digital faces. The code for fine-tuning the model and performing detailed digital face reconstruction is as follows:

For single images: `/Emoface/Standardized_digital_face/gdl_apps/Detail/demos/test_detail_on_images.py`
For video data: `/Emoface/Standardized_digital_face/gdl_apps/Detail/demos/test_detail_on_video.py`