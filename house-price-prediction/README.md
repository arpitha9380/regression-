# Indian House Price Prediction System

This project is a machine learning powered web application for predicting house prices in **Bengaluru, India**.

## Features
- **Dataset:** Trained on the Bengaluru House Price dataset.
- **Model:** Uses Ridge Regression/Gradient Boosting for price estimation in Lakhs.
- **UI:** Modern "Teal & Gold" Indian aesthetic.
- **Modular Code:** Clear separation of data processing, training, and deployment.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download and preprocess data (automatic if you run the app, but optional):
   ```bash
   python house-price-prediction/src/download_data.py
   python house-price-prediction/src/data_preprocessing.py
   ```
3. Run the Flask app:
   ```bash
   python house-price-prediction/app/app.py
   ```
4. Access at `http://127.0.0.1:5000`.

## Tech Stack
- **Backend:** Flask, Scikit-learn, Pandas, Joblib.
- **Frontend:** HTML5, CSS3 (Vanilla), Google Fonts.
- **Model:** Linear Regression, Ridge, Gradient Boosting.
