import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def clean_total_sqft(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def preprocess_data(df, target_col='price'):
    # 1. Cleaning: Remove duplicates and rows with missing price
    df = df.dropna(subset=[target_col])
    df = df.drop_duplicates()
    
    # 2. Extract BHK from size
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else None)
    
    # 3. Handle total_sqft (converting ranges to avg)
    df['total_sqft'] = df['total_sqft'].apply(clean_total_sqft)
    df = df.dropna(subset=['total_sqft'])
    
    # 4. Filter relevant features
    # Keeping it simple for the form: location, total_sqft, bath, bhk, balcony
    relevant_features = ['location', 'total_sqft', 'bath', 'bhk', 'balcony', 'area_type']
    
    # Drop rows with missing values in critical features for training
    df = df.dropna(subset=['location', 'bath', 'bhk', 'area_type'])
    # Impute balcony with 0 if missing
    df['balcony'] = df['balcony'].fillna(0)
    
    X = df[relevant_features]
    y = df[target_col]
    
    # 5. Outlier Removal (Price per sqft is a better metric for Bangalore)
    df['price_per_sqft'] = df['price']*100000 / df['total_sqft'] # Price is in Lakhs
    
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df = remove_pps_outliers(df)
    
    # Final X and y after outlier removal
    X = df[relevant_features]
    y = df[target_col]
    
    # 6. Pipeline for scaling and encoding
    numeric_features = ['total_sqft', 'bath', 'bhk', 'balcony']
    categorical_features = ['location', 'area_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

if __name__ == "__main__":
    raw_path = os.path.join('house-price-prediction', 'data', 'raw', 'bangalore_housing.csv')
    processed_dir = os.path.join('house-price-prediction', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    X, y, preprocessor = preprocess_data(df)
    
    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join('house-price-prediction', 'models', 'preprocessor.pkl'))
    
    # Save processed data (optional but good for debugging)
    processed_df = pd.DataFrame(X_processed)
    processed_df['price'] = y.values
    processed_df.to_csv(os.path.join(processed_dir, 'processed_indian_data.csv'), index=False)
    
    print("Indian data preprocessing complete.")
    print(f"Features: {X.columns.tolist()}")
    print(f"Number of samples: {len(X)}")
