import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PROCESSED_DATA = 'data/processed/train.csv'
OUTPUT_DIR = 'output'

def generate_charts():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(PROCESSED_DATA)
    
    # 1. Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    print("Saved correlation_matrix.png")
    
    # 2. Lab Counts vs Mortality w/ Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='hospital_expire_flag', y='lab_count', data=df)
    plt.title('Total Lab Counts vs Mortality')
    plt.xlabel('Mortality (0=No, 1=Yes)')
    plt.ylabel('Number of Lab Events')
    plt.savefig(os.path.join(OUTPUT_DIR, 'labs_vs_mortality.png'))
    print("Saved labs_vs_mortality.png")

    # 3. Abnormal Labs vs Mortality
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='hospital_expire_flag', y='abnormal_count', data=df)
    plt.title('Abnormal Lab Counts vs Mortality')
    plt.xlabel('Mortality (0=No, 1=Yes)')
    plt.ylabel('Number of Abnormal Labs')
    plt.savefig(os.path.join(OUTPUT_DIR, 'abnormal_labs_vs_mortality.png'))
    print("Saved abnormal_labs_vs_mortality.png")

if __name__ == "__main__":
    generate_charts()
