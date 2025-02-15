# Project Directory Structure
```
├─data
│  ├─interim        # Intermediate data
│  │  │  colors.png
│  │  │  LULC_log.txt
│  │  │  masks.pkl
│  │  │  RSEI_log.txt
│  │  │  SVC_CNN_log.txt
│  │  │  SVC_log.txt
│  │  │  SVC_LSTM_attention_log.txt
│  │  │  SVC_LSTM_log.txt
│  │  │  white.png
│  │  │
│  │  ├─arrays
│  │  │  │  FVC.npz
│  │  │  │  LULC.npz
│  │  │  │  RSEI.npz
│  │  │  │  whole_mask.npz
│  │  │  │
│  │  │  ├─FVC
│  │  │  ├─LULC
│  │  │  └─RSEI
│  │  │
│  │  └─pics
│  │      ├─FVC
│  │      ├─LULC
│  │      └─RSEI
│  │
│  ├─processed      # Processed output data
│  │  └─output_images
│  │
│  └─raw            # Raw data
│      │  New XLSX Worksheet.xlsx
│      ├─FVC
│      ├─LULC
│      └─RSEI
│
├─doc
│      Model_Performance.docx
│      Model_Description.docx
│      Results.docx
│
├─notebooks
│      CnovLSTM.ipynb       # Model training and inference (using Google Colab)
└─src
    │  analyse_res.py       # Computes the proportion of each category in different regions
    │  calculate_outputs_avg.py     # Computes the average values of results
    │  cv_utils.py          # Utility functions for image processing (region segmentation, data transformation, etc.)
    │  extract_mask.py      # Extracts region masks for dataset generation
    │  pic_to_2d_arr.py     # Converts images into 2D arrays
    │  show_color.py        # Displays color maps (for debugging)
    │
    ├─models
    └─visualize
           gbdt_tree_vis.py  # Visualizes GBDT tree structures
           loss_acc_vis.py   # Visualizes model performance metrics
           model_vis.py      # Visualizes model architecture
```

# Data and Training Environment
The dataset and training environment can be accessed at:
[Google Drive](https://drive.google.com/drive/folders/1wMHGfPMLXAOgrnl8m8pGvMwXG7P-Vdjf?usp=drive_link)

# Preprocessing Workflow
1. **Convert Images to 2D Arrays:**  
   Run `src/pic_to_2d_arr.py` to convert all raw images into 2D arrays.
   - These arrays will be saved in `/data/interim/arrays/{map_type}`.
   - Data spanning 20 years will be stored in `/data/interim/arrays/{map_type}.npz`.
   - Full images are named by year, while regional images are named by `year + region ID`.

2. **Extract Useful Mask Regions:**  
   Run `src/extract_mask.py` to extract useful regions from images and save the mask as `/data/interim/arrays/whole_mask.npz`.

3. **Set Up the Training Environment:**  
   - Copy the [training environment](https://drive.google.com/drive/folders/1wMHGfPMLXAOgrnl8m8pGvMwXG7P-Vdjf?usp=drive_link) to your Google Drive.
   - Upload the four `.npz` files (`FVC.npz`, `LULC.npz`, `RSEI.npz`, `whole_mask.npz`) to the **Attention-Convolutional-Neural-Networks-and-Kong-Short-Term_Memory-Model** directory.

# Notebook Directory Structure
```
Vegetation Coverage Prediction
├─models        # Stores trained model weights
│
├─outputs       # Stores output data
│
├─FVC.npz       # 2D arrays from data-interim-arrays
│
├─LULC.npz      # 2D arrays from data-interim-arrays
│
├─RSEI.npz      # 2D arrays from data-interim-arrays
│
└─whole_mask.npz # Mask file from data-interim-arrays
```

# Notebook Execution Steps
1. **Mount Google Drive:**  
   - Run the `mount` cell to attach the directory.

2. **Select Model and Map Type:**  
   - Run `select model and map` to choose the model and dataset.

3. **Load Data:**  
   - Run `load data` to load the selected map type and generate the dataset.

4. **Load Model Classes:**  
   - Execute all cells in the `Models` section.

5. **Train the Model:**  
   - Run `Train-AMP Train` to accelerate model training.

6. **Evaluate Performance:**  
   - After training, run `Test-Test All` to generate model performance reports, including **ROC, PR curves, etc.**.

7. **Generate Future Prediction Maps:**  
   - Run `graphs` to visualize and generate future forecast maps.


