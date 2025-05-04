# NYCU Computer Vision 2025 Spring HW3
- StudentID: 313553044
- Name: 江仲恩

## Introduction
In this assignment, we developed a robust cell instance segmentation model capable of accurately classifying and segmenting cells in microscopy images from the HW3 dataset. To enhance the model’s generalization ability, we adopted the Mask R-CNN architecture with Swin-Tiny as the backbone, a Region Proposal Network (RPN), a Path Aggregation Network (PAN), and dedicated classification and segmentation heads.
Given the limited size of the dataset, we applied **Repeated Augmentation** during training to effectively increase the diversity of training samples.
To further boost inference performance, we introduced multiple Test-Time Augmentation (TTA) strategies, including horizontal and vertical flips, 90°/180° rotations, brightness and contrast adjustments, hue shifts, Gaussian blur, and transposition. Each augmented image was transformed back to the original space and aggregated, followed by Non-Maximum Suppression (NMS) to remove redundant predictions and retain the most confident ones.
Our best configuration achieved a mAP of 0.**4488**, demonstrating the overall effectiveness of our model architecture, training strategy, and inference-time enhancements.

## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW3.git
cd NYCU-Computer-Vision-2025-Spring-HW3
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW3
├── hw3-data-release
│   ├── train
│   ├── test_release
│   └── test_image_name_to_ids.json
├── environment.yml
├── main.py
├── train.py
├── test.py
.
.
.
```

4. Run for Train
    1. Train Model 
    ```
    python main.py DATAPATH [--epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--save_path SAVE_FOLDER] [--mode train]
    ```
    Example
    ```
    python main.py ./hw3-data-release --epochs 30 --batch_size 2 --learning_rate 1e-4 --decay 1e-4 --save_path saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```
    Example
    ```
    python main.py ./hw3-data-release --mode test
    ```

## Performance snapshot
### Training Parameter Configuration

| Parameter        | Value                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------|
| Pretrained Weight| Swin_V2_T_Weights                                                                                       |
| Learning Rate    | 0.0001                                                                                                  |
| Batch Size       | 2                                                                                                       |
| Epochs           | 30                                                                                                      |
| decay            | 0.005                                                                                                   |
| Optimizer        | AdamW                                                                                                   |
| Eta_min          | 0.000001                                                                                                |
| T_max            | 30                                                                                                      |
| Scheduler        | `CosineAnnealingLR`                                                                                     |
| Criterion        | `CrossEntropyLoss(Classification)` + `Smooth L1 Loss(Localization)` + `Binary Cross Entropy Loss (Mask)`|

<!-- ### Training Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/training_curve.png)
### validate mAP Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/val_map_curve.png)
### validate AP / AR Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/ResNet50_Original.png) -->

### Performance
|                  | mAP                      |
|------------------|--------------------------|
| Validation       | 0.3426                   |
| Public Test      | 0.4488                   |

