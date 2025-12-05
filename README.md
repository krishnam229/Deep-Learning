# **Deep Learning–Based Prediction of NYC Yellow Taxi Trip Duration Using Weather Data**

This project (CS672 – Introduction to Deep Learning, Fall 2025) focuses on predicting **NYC Yellow Taxi trip duration** using Deep Learning regression models. The dataset combines **TLC Taxi trip records** with **NYC weather data** to improve prediction accuracy through feature enrichment. Multiple neural network architectures are implemented and compared to determine the best-performing model.

---

## **📌 Project Objective**

- Merge NYC Yellow Taxi data with January 2020 weather data from Meteostat  
- Preprocess, clean, and scale the combined dataset  
- Build and train multiple TensorFlow regression models  
- Compare model performance using MSE and MAE  
- Select the best model and generate predictions  

This project strictly uses **neural network–based models**.

---

## **📂 Datasets Used**

### **1. NYC Yellow Taxi Trip Data**
Includes:
- Pickup & dropoff timestamps  
- Distance traveled  
- Passenger count  
- Location IDs  
- Vendor/payment details  
- Fare information  

### **2. NYC Weather Data (January 2020 – Meteostat)**

Contains:
- Average, minimum, maximum temperature  
- Precipitation  
- Snowfall  
- Wind speed  
- Barometric pressure  
- Sunshine duration  

Weather data is merged by **date** to align with taxi pickup timestamps.

---

## **🧹 Data Preprocessing**

- Converted timestamps into meaningful features (hour, day, weekday)  
- Removed invalid and extreme trip durations  
- Scaled numerical features using **StandardScaler**  
- Merged taxi trip data with corresponding daily weather metrics  
- **80/20 time-aware split**: first 80% for training, last 20% for validation  

---

## **🤖 Deep Learning Models Implemented**

### **1. Linear Regression Model (TensorFlow Sequential)**

- No hidden layers  
- Baseline comparison model  

### **2. Multi-Layer Perceptron (MLP)**

- One or more dense hidden layers  
- ReLU activation  
- Fully connected architecture  

### **3. Deep Neural Network (DNN)**

- Two or more hidden layers  
- Dropout for regularization  
- Best for capturing nonlinear patterns  

---

## **⚙️ Training Configuration**

All models were trained with:

- **Loss functions:**  
  - Mean Squared Error (MSE)  
  - Mean Absolute Error (MAE)

- **Optimizers tested:**  
  - SGD  
  - Adam  
  - RMSprop

- **Learning Rates:**  
  - 0.001  
  - 0.01  
  - 0.0001  

- **Epochs:** 100  
- **Batch size:** 32 (default)  

Training vs. validation loss curves were plotted for each experiment.

---

## **📊 Model Comparison & Results**

Models were evaluated using:

- Validation MSE  
- Validation MAE  
- Training stability  
- Overfitting behavior  
- Learning rate sensitivity  

**The best-performing model** (based on lowest validation MAE) was:

### ✅ *DNN with RMSprop (lr = 0.001)*  

> Update this line if your own experiment results differ.

This model demonstrated strong generalization and stable learning during training.

---

## **🧪 Predictions**

Once trained, predictions were generated using:

```python
model.predict()


## 📈 Results Summary (Sample)

| Model | Optimizer | LR | Val MSE | Val MAE |
|-------|-----------|----|---------|---------|
| Linear Regression | Adam | 0.001 | 0.82 | 0.31 |
| MLP | RMSprop | 0.001 | 0.56 | 0.24 |
| DNN | RMSprop | 0.001 | **0.32** | **0.19** |
