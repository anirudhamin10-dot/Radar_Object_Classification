# Pattern Recognition Project: Radar Classification

This project classifies radar objects using two machine learning algorithms: Support Vector Machine (SVM) and Random Forest.

## Performance Notes

Performance metrics were observed on a laptop (Ryzen 5 5600H, 16GB RAM, RTX 3050 laptop GPU 50W) for reference:

- **With 5 sequences:** The code completes execution in approximately 1 minute. Memory usage is minimal (typically <1GB).
- **With full 158 sequences:** The complete execution time is approximately 6-7 minutes on the same hardware configuration. Memory usage is minimal (typically <1GB).

## How to Run

### 1. Install Libraries

All required Python libraries are listed in the `requirements.txt` file. Install them using pip:
```bash
pip install -r requirements.txt
```

### 2. Get Data

The project requires the Radar Scenes dataset, which is provided via a shared Google Drive folder for convenient access.

1. Access the dataset through the Google Drive link: [Radar Scenes Dataset](https://drive.google.com/drive/folders/12_-l2qHmVRkHxGsDgGtbTfbKU65T3HFg?usp=drive_link)
> **Note:** This dataset contains only the first 5 of the original 158 sequences from the Radar Scenes dataset.
2. In the root directory of this project, create a new folder named `data`.
3. Place the downloaded dataset sequence folders (which contain the necessary `radar_data.h5` files) directly inside the newly created `data` folder.
4. For all 158 sequences download the original dataset from `https://zenodo.org/records/4559821`.

#### Directory Structure Example:
```
ProjectRoot/
├── Main.py
├── requirements.txt
└── data/
    ├── sequence_0000/
    │   └── radar_data.h5
    ├── sequence_0001/
    │   └── radar_data.h5
    └── ...
```

### 3. Run Code

Execute the main script from your terminal:
```bash
python Main.py
```

The script will handle data loading, feature extraction, model training (SVM and Random Forest), and evaluation.

## Project Overview

### Algorithms Used

- **Support Vector Machine (SVM):** A supervised learning algorithm that finds the optimal hyperplane for classification.
- **Random Forest:** An ensemble learning method that constructs multiple decision trees for robust classification.

### Dataset

This project uses the [Radar Scenes dataset](https://radar-scenes.com/), which contains radar data for object detection and classification tasks.

## Requirements

See `requirements.txt` for a complete list of dependencies. Key libraries include:

- NumPy
- scikit-learn
- h5py
- pandas

## License

Please refer to the [Radar Scenes dataset license](https://radar-scenes.com/) for data usage terms.

## Acknowledgments

This project uses the Radar Scenes dataset. For more information about the dataset, visit [https://radar-scenes.com/](https://radar-scenes.com/).
