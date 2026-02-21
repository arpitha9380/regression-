# 🇮🇳 Indian House Price Prediction System

**Indian House Price Prediction System** is a high-end machine learning application designed to estimate property prices in **Bengaluru, India**. Built with a focus on local market dynamics, it features an "Indian Modern" aesthetic and a robust data pipeline.

![Project Preview](https://via.placeholder.com/800x400.png?text=Indian+House+Price+Prediction+System+Dashboard) 
*Note: Replace this with your actual application screenshot.*

## 🌟 Key Features

- **Bengaluru Focused:** Specifically trained on the Bengaluru House Price dataset to capture city-specific trends.
- **Advanced Preprocessing:** Handles complex real estate data (e.g., BHK extraction, square feet ranges, location-based outlier removal).
- **Multi-Model Pipeline:** Compares Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.
- **"Indian Modern" UI:** A premium user interface using Teal and Gold accents, optimized for better user experience.
- **Modular Architecture:** Clean separation of concerns between data processing, model training, and web deployment.
- **Dockerized:** Ready for containerized deployment in various environments.

## 🛠️ Tech Stack

- **Backend:** Python, [Flask](https://flask.palletsprojects.com/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Joblib](https://joblib.readthedocs.io/)
- **Frontend:** HTML5, CSS3 (Vanilla), Google Fonts (Montserrat)
- **Visualization:** Matplotlib, Seaborn
- **Containerization:** Docker

## 📂 Project Structure

```text
├── app/                        # Flask Web Application
│   ├── static/                 # CSS, Images, JS
│   ├── templates/              # HTML Templates
│   └── app.py                  # Web application entry point
├── data/                       # Dataset storage (Raw & Processed)
├── models/                     # Saved ML models (Joblib format)
├── notebooks/                  # Jupyter notebooks for experimentation
├── src/                        # Core Data & ML pipeline
│   ├── data_preprocessing.py   # Cleaning and engineering
│   ├── download_data.py        # Automated data acquisition
│   ├── eda.py                  # Exploratory Data Analysis
│   └── train.py                # Model training and selection
├── Dockerfile                  # Containerization configuration
├── requirements.txt            # Python dependencies
└── report.md                   # Detailed model performance report
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip (Python Package Installer)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arpitha9380/project.git
   cd regression-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r house-price-prediction/requirements.txt
   ```

3. **Prepare the data (Optional but recommended):**
   ```bash
   # Download city-specific data
   python house-price-prediction/src/download_data.py
   # Clean and preprocess
   python house-price-prediction/src/data_preprocessing.py
   ```

4. **Run the application:**
   ```bash
   # From the house-price-prediction directory
   python house-price-prediction/app/app.py
   ```
   Access the UI at `http://127.0.0.1:5000`.

### Running with Docker

```bash
docker build -t house-price-app .
docker run -p 5000:7860 house-price-app
```

## 📊 Model Performance

Our pipeline evaluates multiple models to ensure the best fit for the volatile Bengaluru market.

| Model | RMSE (Lakhs) | R² Score |
|-------|--------------|----------|
| **Ridge Regression** | **58.79** | **0.77** |
| Linear Regression | 58.88 | 0.77 |
| Random Forest | 63.76 | 0.73 |
| Gradient Boosting | 64.99 | 0.72 |

*The system defaults to using **Ridge Regression** for predictions due to its stability across various property types.*

## 🛣️ Roadmap

- [ ] Support for other Indian cities (Mumbai, Delhi, Pune).
- [ ] Integration with real-time property APIs.
- [ ] Advanced deep learning models for better accuracy.
- [ ] Interactive maps for location-based price visualization.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
