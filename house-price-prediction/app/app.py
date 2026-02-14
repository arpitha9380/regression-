from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import sys

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'house_price_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.pkl')

# Load model and preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Get locations from preprocessor (if available)
try:
    # Assuming location is the first categorical transformer
    onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # The columns in preprocessor are [num_features, cat_features]
    # We need to find the locations
    locations = sorted([c.replace('x0_', '') for c in onehot.get_feature_names_out() if c.startswith('x0_')])
except:
    locations = ["Electronic City Phase II", "Chikka Tirupathi", "Uttarahalli", "Marathahalli", "Whitefield"]

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'location': request.form.get('location'),
            'total_sqft': float(request.form.get('total_sqft')),
            'bath': float(request.form.get('bath')),
            'bhk': int(request.form.get('bhk')),
            'balcony': float(request.form.get('balcony')),
            'area_type': request.form.get('area_type')
        }
        
        # Convert to DataFrame
        df_input = pd.DataFrame([data])
        
        # Preprocess
        X_processed = preprocessor.transform(df_input)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        # Result in Lakhs
        return render_template('index.html', 
                               prediction_text=f'Estimated Price: ₹{prediction:,.2f} Lakhs',
                               form_data=request.form,
                               locations=locations)
    except Exception as e:
        return render_template('index.html', 
                               error=f"Error: {str(e)}", 
                               form_data=request.form,
                               locations=locations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
