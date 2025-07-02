# Random-Forest-Based-Land-Cover-Classification-Using-Sentinel-2-Remote-Sensing-Data-in-Python
This repository demonstrates a complete land cover classification pipeline using Sentinel-2 multispectral imagery and a Random Forest machine learning model in Python. 
The workflow includes preprocessing, index calculation, training using labeled samples, pixel-wise classification, and accuracy assessment.

📌 Overview:
    This project presents a reproducible machine learning pipeline for Land Use/Land Cover (LULC) classification using harmonized Sentinel-2 imagery. It leverages both raw spectral bands and derived indices (NDVI, MNDWI, NBI) to build a robust Random Forest classifier. The trained model distinguishes five major land cover types with high accuracy using six high-resolution bands and three spectral indices.

🖥️ Tools & Technologies:
    Language: Python (Jupyter Notebook)
    Model: Random Forest (scikit-learn)
    Satellite Data: Sentinel-2 (Harmonized), Bands B2–B12

🛠️ Key Features:
    ✅ Multiband raster processing and normalization (0–1 scaling)
    ✅ NDVI, MNDWI, and NBI index computation
    ✅ Training sample extraction from shapefile vector layers
    ✅ Feature stacking: 6 bands + 3 indices
    ✅ 80:20 train-test split
    ✅ Model trained using 100 epochs
    ✅ Pixel-wise classification of entire raster
    ✅ Accuracy assessment with metrics and visualization

📚 Libraries Used:
| Library        | Purpose                                   |
| -------------- | ----------------------------------------- |
| `rasterio`     | Read/write geospatial raster data         |
| `geopandas`    | Load and process vector data (shapefiles) |
| `numpy`        | Numerical operations                      |
| `pandas`       | DataFrame manipulation                    |
| `scikit-learn` | Machine learning and model evaluation     |
| `matplotlib`   | Plotting graphs and outputs               |
| `seaborn`      | Visualizing confusion matrix              |
| `joblib`       | Saving/loading trained models             |


1️⃣ Data Acquisition:
    Input: Harmonized Sentinel-2 imagery
    Required Bands: Blue, Green, Red, NIR, SWIR1, SWIR2
    Example Path: E:/RandomeForest/S2_HARMONIZED_20220307.tif

2️⃣ Reflectance Scaling:
    Apply 2nd–98th percentile scaling to normalize pixel values
    Final range: 0 to 1

3️⃣ Spectral Index Computation:
| Index     | Formula                           | Interpretation    |
| --------- | --------------------------------- | ----------------- |
| **NDVI**  | (NIR − Red) / (NIR + Red)         | Vegetation health |
| **MNDWI** | (Green − SWIR1) / (Green + SWIR1) | Water bodies      |
| **NBI**   | (SWIR1 − NIR) / (SWIR1 + NIR)     | Built-up areas    |


4️⃣ Training Sample Preparation:
    Classes:
      Built-up
      Water
      Vegetation
      Barren
      Agriculture 
   Samples:
      30 points per class = 150 total labeled points
  Features:
      6 spectral bands + 3 indices = 9 total features
  Split:
    Training set = 80%
    Testing set = 20%
    Implemented using train_test_split() from scikit-learn

5️⃣ Model Training:
    Classifier: Random Forest
    n_estimators: 100
    Epochs: Model trained using 100 iterations (shuffled sample batches)
    Output Model: RF_model.pkl
    
6️⃣ Classification:
    Trained model applied across entire raster
    Output: Thematic land cover classification map as GeoTIFF (classified_map.tif)
    Output Model: RF_model.pkl

7️⃣ Accuracy Assessment:
    Evaluated using 20% held-out test data
    Metrics:
      Confusion Matrix
      Precision, Recall, F1-score
      Overall Accuracy
    Output: confusion_matrix.png

🗂️ Output Files:
| File Name              | Description                               |
| ---------------------- | ----------------------------------------- |
| `RF_model.pkl`         | Trained Random Forest model               |
| `classified_map.tif`   | Final land cover classification map       |
| `confusion_matrix.png` | Accuracy visualization (confusion matrix) |

🔗 Data Sources:
    📥 Copernicus Open Access Hub – Sentinel-2 imagery
    📥 Bhoonidhi ISRO Portal – Indian EO datasets (e.g., LISS-IV)
    🌐 Google Earth Engine (GEE) – Cloud-based geospatial data and processing

🎯 Conclusion:
    This project delivers an effective, high-accuracy workflow for land cover classification using remote sensing and machine learning in Python. By combining Sentinel-2 spectral information with vegetation, water, and built-up indices, and training a Random Forest classifier over 100 epochs, the methodology supports real-world LULC mapping applications.
