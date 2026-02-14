# Indian House Price Prediction System Report

## 1. Introduction
This project implements an end-to-end machine learning pipeline to predict house prices in Bengaluru, India. It includes automated data acquisition, cleaning, preprocessing, and a web-based prediction interface.

## 2. Dataset Overview
- **Source:** Bengaluru House Data (Kaggle/GitHub)
- **Features:** Location, Total Square Feet, BHK, Bathrooms, Balcony, Area Type.
- **Target:** Price (in Lakhs)

## 3. Exploratory Data Analysis (EDA)
EDA focused on price per square foot across different locations and area types. Extreme outliers in the Bengaluru market (e.g., highly inflated prices or incorrect square footage) were handled to stabilize the model.

## 4. Preprocessing
- **BHK Extraction:** Converted "2 BHK", "3 Bedroom" into numeric values.
- **Area Cleaning:** Handled square feet ranges (e.g., "1200 - 1400") by averaging.
- **Outlier Removal:** Used location-based price-per-sqft filtering (Mean ± 1 StdDev).
- **Encoding/Scaling:** Standard Scaling for numeric features and One-Hot Encoding for locations/area types.

## 5. Model Performance
| Model | RMSE (Lakhs) | R² Score |
|-------|--------------|----------|
| Linear Regression | 58.88 | 0.77 |
| Ridge Regression | 58.79 | 0.77 |
| Random Forest | 63.76 | 0.73 |
| Gradient Boosting | 64.99 | 0.72 |

The **Ridge Regression** model provided the most stable results for the Bengaluru market, followed by Linear Regression.

## 6. UI Redesign
The interface was redesigned with an **"Indian Modern"** aesthetic:
- **Colors:** Teal (#008080) and Gold (#d4af37).
- **Typography:** Montserrat for a sleek, professional look.
- **User Experience:** Focus on clear labels for BHK, Square Feet, and location-based selections.

## 7. Conclusion
The system provides a robust estimation tool for the Bengaluru real estate market, wrapped in a culturally resonant and premium user interface.
