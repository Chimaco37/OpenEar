# Squirrel
Low-cost, high-throughput and accurate maize ear phenotyping system
![logo](https://github.com/user-attachments/assets/1392e8f6-083a-4b8b-8c88-b227d3edfdba)

## Features
- **Graphical User Interface (GUI):** User-friendly interface for users without programming expertise.
- **Command Line Interface (CLI):** Direct use via command line.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Chimaco37/Squirrel.git
    ```
2. **Install dependencies (Only when you need to use CLI):**

    ```bash
    cd Squirrel/
    pip install -r requirements.txt
    ```

3. **Download necessary models:**  

    First, download the required model files from the [Models Figshare Repository](https://doi.org/10.6084/m9.figshare.26282731).
    
    **Placement of model files:**
    
    - **For the GUI of the ear phenotyping system:**
    
      Place the downloaded model files in the specified directory with the following steps:

        ```bash
        cd ear/GUI/
        unzip Models.zip
        cp Models/Ear_Models/* models/
        ```
    
    - **For Command Line Interface (CLI) usage:**
   
      You can place the models in any location that is convenient for you.

4. **Download GUIs:**

    Download the GUI files from the [GUIs Figshare Repository](https://doi.org/10.6084/m9.figshare.26363107).

    - **Placement of GUI files:**
   
      After downloading, place the GUI files in the respective ./GUI directory with these steps:
  
        ```
        cd leaf/GUI/
        unzip GUIs.zip
        cp GUIs/Lizard.exe ./
        ```

## GUI Usage

### 🐿️The 'Squirrel' System
![image](https://github.com/user-attachments/assets/b7045c19-be7b-40f4-835d-b8b99b7ed893)

- **Video Process:**  
  Click the "Video Process" button, choose the video folder and the folder where images will be saved.  
  The system will process these videos into projections and ear images.

- **Model inference:**  
  Click the "Model Inference" button, select the image folder and the results output folder.  
  The images, including projections and ear images, will be analyzed through model inference to generate results.


### 🐿️The 'Squirrel' System

- **Video Preprocessing:**
```
python Convert_videos_to_projections.py -v VIDEO_FOLDER -p PARAMETER_FOLDER -o OUTPUT_PATH -c CORES_NUMBER -i PYTHON_INTERPRETER

optional arguments:
  -v: Path to the original video folder (default is ./videos/)
  -p: Path to the image undistortion parameter folder (default is ./image_process/)
  -o: Output undistorted image folder (default is ./undistorted/)
  -c: Number of cores used for parallel processing (default is 5)
  -i: Path to your python interpreter
```

- **Model training for kernel-related and ear-related traits:**

```bash
yolo segment train data=/path/to/your/projection/dataset/data.yaml model=/path/to/your/projection/model.pt epochs=200 batch=4 patience=30 device=0,1,2,3 name=projection_model_training

yolo segment train data=/path/to/your/ear/dataset/data.yaml model=/path/to/your/ear/model.pt epochs=200 batch=32 patience=30 device=0 name=ear_model_training
```

- **Model inference for kernel-related and ear-related traits:**

```
yolo segment predict model=projection.pt source=/path/to/projection/image/folder/ device=cpu conf=0.25 iou=0.4 show_labels=False save_txt=True show_conf=False boxes=False imgsz=1600 max_det=1000 retina_masks=True  name=projection

yolo segment predict model=models/Ear.pt source=/data1/fanshaoqi/dataset/ear_base_mark_24_11_27/ear device=0 conf=0.5 imgsz=1440 show_labels=False show_conf=False boxes=True max_det=1 save_txt=True retina_masks=True name=prediction project=/data1/fanshaoqi/dataset/ear_base_mark_24_11_27
```
- **Output analysis:**
```
python Model_output_analysis.py -i PROJECTION_IMAGE_FOLDER -e EAR_LABEL_FOLDER -p PROJECTION_LABEL_FOLDER -o OUTPUT_PATH -m MODEL_PATH

optional arguments:
  -i: Path to the projection image folder (default is ./images/projection/)
  -e: Path to the ear model output label folder (default is ./result/ear/labels/)
  -p: Path to the projection model output label folder (default is ./result/projection/labels/)
  -o: Analyzed results output folder (default is ./)
  -m: CNN model path folder (default is ./models/)
```
