<img src="https://github.com/user-attachments/assets/18c3a99d-f414-4d89-85e7-2900d2b135ea" alt="ear_logo" height="120px" />

# OpenEar
Open-source, low-cost, high-throughput and accurate maize ear phenotyping system

## Features
- **Graphical User Interface (GUI):** User-friendly interface for users without programming expertise.
- **Command Line Interface (CLI):** Direct use via command line.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Chimaco37/OpenEar.git
    ```
2. **Install dependencies (Only when you need to use CLI):**

    ```bash
    cd OpenEar/
    pip install -r requirements.txt
    ```

3. **Download necessary models:**  

    First, download the required model files from the [Models Figshare Repository](https://doi.org/10.6084/m9.figshare.29115983).
    
    **Placement of model files:**
    
    - **For the GUI of the ear phenotyping system:**
    
      Place the downloaded model files in the specified directory with the following steps:

        ```bash
        unzip Squirrel_Models.zip
        cp Squirrel_Models/* models/
        ```
    
    - **For Command Line Interface (CLI) usage:**
   
      You can place the models in any location that is convenient for you.

4. **Download GUIs:**

    Download the GUI files from the [GUIs Figshare Repository](https://doi.org/10.6084/m9.figshare.26363107).

    - **Placement of GUI files:**
   
      After downloading, place the GUI files in the respective ./GUI directory with these steps:
  
        ```
        mv Squirrel.exe ./Squirrel
        ```

## GUI Usage
### 🐿️ OpenEar
![Squirrel](https://github.com/user-attachments/assets/a23fc0f7-b28a-4154-84b8-7a2faa5af0db)

## Key Functions

### 1. Video Processing
- **Purpose:** Unroll the maize ear surface by processing video into projection image and extract frames for ear-level traits extraction.
- **Usage:**
  1. Click the **Video Process** button.
  2. **Confirm the operation** when prompted.
  3. Select the following folders:
     - **Input Videos:** Folder containing raw video files (e.g., `.mp4`, `.avi`).
     - **Output image folder:** Destination folder for the generated projection and ear images.
     - **Model folder:** Folder that contains the image process parameters folder. 
  4. Specify the **number of threads** based on your device's capabilities.
  5. The system processes the videos accordingly.

### 2. Model Inference
- **Purpose:** Run the model to analyze projection and ear images, then generate phenotypic measurements.
- **Usage:**
  1. Click the **Model Inference** button.
  2. **Confirm the operation** when prompted.
  3. Choose the **device** for inference (GPU or CPU).
  4. Select the following folders:
     - **Image folder:** Folder containing the projection and ear images folder.
     - **Model Folder:** Directory where the model files are stored.
     - **Output Folder:** Destination for the generated phenotypic data.
  5. The system then runs inference and outputs the phenotypic measurements.
  
---

### 3. Manual Adjustment of Results
- **Purpose:** Fine-tune model predictions by manually adjusting key phenotypic positions on images.
- **Features:**
  - **Colored Lines Indicate:**
    - **Green:** Ear rotation boundary markers
    - **Red:** Kernel row indicator
- **Usage:**
  - **Drag:** Use the **left mouse button** to drag and adjust **boundary markers**, avoid error introduced by inprecise boundary detection during video processing.
  - **Adjust Kernel Row:** **Double-click** the **left mouse button** to add or subtract kernel row number (depending on whether a kernel row indicator exists at the clicked location).
  - **Change Data Availability:** **Single-click** the **right mouse button** to change the data availability for the poorly developed ear (all availabel/kernel row 'NA'/all traits 'NA').

---

## CLI Usage
- **Video processing:**
```
python 2_Video_processing.py -v VIDEO_FOLDER -m MODEL_FOLDER -o OUTPUT_PATH -c CORES_NUMBER

optional arguments:
  -v: Path to the original video folder (default is ./videos/)
  -m: Path to the model folder (default is ./models/)
  -o: Output image folder (default is ./images/)
  -c: Number of cores used for parallel processing (default is 5)
  -e: Ear center in the image used for video processing (default is 0.5)
```
                        
- **Model training for kernel-related and ear-related traits:**

```bash
yolo segment train data=/path/to/projection/dataset/data.yaml model=/path/to/model.pt epochs=200 batch=4 patience=30 device=0,1,2,3 name=projection_model_training

yolo segment train data=/path/to/ear/dataset/data.yaml model=/path/to/model.pt epochs=200 batch=32 patience=30 device=0 name=ear_model_training
```

- **Model inference for kernel-related and ear-related traits:**

```
yolo segment predict model=Projection.pt source=/path/to/projection/folder/ device=cpu conf=0.25 iou=0.4 show_labels=False save_txt=True show_conf=False show_boxes=False imgsz=1600 max_det=1000 retina_masks=True  name=projection

yolo segment predict model=Ear.pt source=/path/to/ear/folder/ device=0 conf=0.5 imgsz=640 show_labels=False show_conf=False show_boxes=True max_det=1 save_txt=True retina_masks=True name=ear
```

- **Model output analysis:**
```
python 3_Model_output_analysis.py -i PROJECTION_IMAGE_FOLDER -e EAR_LABEL_FOLDER -p PROJECTION_LABEL_FOLDER -o OUTPUT_PATH -m MODEL_PATH

optional arguments:
  -i: Path to the projection image folder (default is ./images/projection/)
  -e: Path to the ear model output label folder (default is ./output/ear/labels/)
  -p: Path to the projection model output label folder (default is ./output/projection/labels/)
  -o: Analyzed results output folder (default is ./output/)
  -m: CNN model path folder (default is ./models/)
```
