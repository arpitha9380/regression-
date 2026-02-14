import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda(raw_path, output_dir):
    df = pd.read_csv(raw_path)
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of SalePrice
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='blue')
    plt.title('Distribution of Sale Price')
    plt.savefig(os.path.join(output_dir, 'saleprice_dist.png'))
    plt.close()
    
    # 2. Correlation Heatmap (only numeric)
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Scatter Plot: GrLivArea vs SalePrice
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, alpha=0.5)
    plt.title('Living Area vs Sale Price')
    plt.savefig(os.path.join(output_dir, 'grlivarea_vs_saleprice.png'))
    plt.close()
    
    # 4. Boxplot: Neighborhood vs SalePrice
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
    plt.xticks(rotation=90)
    plt.title('Neighborhood vs Sale Price')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_vs_saleprice.png'))
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    raw_data_path = os.path.join('house-price-prediction', 'data', 'raw', 'ames_housing.csv')
    plots_dir = os.path.join('house-price-prediction', 'notebooks', 'plots')
    generate_eda(raw_data_path, plots_dir)
