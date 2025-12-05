# Deep Learning–Based Prediction of NYC Yellow Taxi Trip Duration Using Weather Data

This repository contains two major projects completed for **CS672 – Introduction to Deep Learning (Fall 2025)** at Pace University:

1. **Exploratory Data Analysis (EDA) of NYC Yellow Taxi Trips – Project 1**
2. **Deep Learning Regression Models for Trip Duration Prediction – Project 2**

The goal is to understand factors influencing taxi ride durations and build neural network models that accurately predict trip time using enriched features, including NYC weather conditions.

---

# 1. Exploratory Data Analysis (Project 1)

This notebook performs comprehensive **Exploratory Data Analysis (EDA)** on NYC Yellow Taxi Trip Records from 2020.

### Objectives
- Create a **training dataset** using March 2020 trips and an **evaluation dataset** using May 2020 trips.
- Clean and prepare data by handling missing values, nulls, NaNs, and outliers.
- Transform all categorical and date-time features into numerical formats suitable for modeling.
- Understand data types and distributions (numeric and categorical).
- Generate visual and statistical summaries using:
  - **pandas**, **numpy**, **matplotlib**, **seaborn**
  - **TensorFlow Data Validation (TFDV)** + Apache Beam for feature statistics
- Identify significant features influencing **trip duration** and ride-time prediction.

### Key Outcomes
- A fully cleaned and processed dataset.
- Insights into variable relationships and trends.
- Prepared foundation for modeling used in Project 2.

---

# 2. Deep Learning Regression for Trip Duration Prediction (Project 2)

This project builds and compares multiple neural network architectures to predict NYC Yellow Taxi trip duration using both **trip features** and **weather data**.

### Steps Performed
- Retrieved NYC climate data for January 2020 from Meteostat.
- Merged weather data with taxi records on pickup date.
- Scaled numeric features using StandardScaler.
- Trained three types of neural network models:
  1. **Linear Regression (TF Sequential, no hidden layers)**
  2. **Multi-Layer Perceptron (MLP)**
  3. **Deep Neural Network (DNN)** with ≥2 hidden layers and dropout

### Training Configuration
- **Loss Functions:** MSE, MAE  
- **Optimizers:** SGD, Adam, RMSprop  
- **Learning Rates:** 0.01, 0.001, 0.0001  
- **Epochs:** 100  
- **Batch Size:** 32  

Training and validation loss curves are plotted to evaluate model performance.

### Best Model
Based on validation MAE:

> **DNN with RMSprop (lr = 0.001)**  
> Showed the most stable and accurate predictions.

*(Replace this if your own results differ.)*

---

# 3. Repository Structure

```text
📦 Deep-Learning
│
├── NYC_Taxi_Project1_EDA.ipynb              # Project 1 EDA Notebook
├── NYC_Taxi_Trip_Duration_DeepLearning.ipynb # Project 2 DL Model Notebook
├── saved_models.zip                          # Trained model files (optional)
└── README.md                                 # Documentation
```

---

# 4. Environment & Dependencies

This project uses:

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

Install required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Kernel used during development: **Python (dl2020) – Anaconda environment**

---

# 5. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnam229/Deep-Learning.git
   ```
2. Open the notebooks in **Jupyter Notebook** or **VS Code**.
3. Ensure required libraries are installed.
4. Run all cells sequentially to reproduce the analysis and results.

---
