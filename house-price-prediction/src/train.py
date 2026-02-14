import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from data_preprocessing import preprocess_data

def train_models(processed_data_path):
    # Load raw data and re-run preprocessing to get X and y properly
    raw_path = os.path.join('house-price-prediction', 'data', 'raw', 'bangalore_housing.csv')
    df = pd.read_csv(raw_path)
    X, y, preprocessor = preprocess_data(df, target_col='price')
    
    # We need to transform X before splitting or include it in a pipeline
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        rmse_cv = -cv_scores.mean()
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'CV RMSE': rmse_cv,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
    return pd.DataFrame(results), X_train, y_train, X_test, y_test

def hyperparameter_tuning(X_train, y_train):
    print("\nStarting Hyperparameter Tuning...")
    
    # Gradient Boosting Tuning
    param_grid_gb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    gb = GradientBoostingRegressor(random_state=42)
    grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search_gb.fit(X_train, y_train)
    
    print(f"Best GB Params: {grid_search_gb.best_params_}")
    return grid_search_gb.best_estimator_

if __name__ == "__main__":
    processed_data_path = os.path.join('house-price-prediction', 'data', 'processed', 'processed_data.csv')
    df_results, X_train, y_train, X_test, y_test = train_models(processed_data_path)
    
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Final Evaluation of best tuned model
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nBest Tuned Model (Gradient Boosting) Final Results - RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    # Save the model
    joblib.dump(best_model, os.path.join('house-price-prediction', 'models', 'house_price_model.pkl'))
    print("Best model saved to models/house_price_model.pkl")
    
    # Save results summary
    df_results.to_csv(os.path.join('house-price-prediction', 'models', 'model_comparison.csv'), index=False)
