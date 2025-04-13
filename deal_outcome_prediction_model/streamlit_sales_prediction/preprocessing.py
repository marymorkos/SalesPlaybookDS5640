import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def load_model_components():
    """Load the model and preprocessing components"""
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/encoding_info.pkl', 'rb') as f:
        encoding_info = pickle.load(f)
    
    return model, scaler, encoding_info

def preprocess_data(input_data, scaler, encoding_info):
    """Preprocess input data for model prediction"""
    # Make a copy of the input data
    df = input_data.copy()
    
    # Calculate engineered features
    # 1. Company Age
    current_year = datetime.now().year
    df['Company_Age'] = current_year - df['Year Founded']
    df.drop('Year Founded', axis=1, inplace=True)
    
    # 2. Contact Frequency
    df['Contact_Frequency'] = np.where(
        df['Days to close'] > 0,
        df['Number of times contacted'] / df['Days to close'],
        df['Number of times contacted']
    )
    
    # 3. Revenue per Employee
    df['Revenue_per_Employee'] = df['Annual Revenue'] / (df['Number of Employees'] + 1)
    
    # 4. Submission Conversion Rate
    df['Submission_Conversion_Rate'] = np.where(
        df['Number of Sessions'] > 0,
        df['Number of Form Submissions'] / df['Number of Sessions'],
        0
    )
    
    # 5. Page Depth
    df['Page_Depth'] = np.where(
        df['Number of Sessions'] > 0,
        df['Number of Pageviews'] / df['Number of Sessions'],
        0
    )
    
    # Identify numerical columns for scaling
    numerical_columns = [
        'Annual Revenue', 'Number of times contacted', 'Amount in company currency',
        'Amount', 'Forecast amount', 'Number of Pageviews', 'Number of Employees',
        'Number of Sessions', 'Days to close', 'Number of Form Submissions',
        'Company_Age', 'Contact_Frequency', 'Revenue_per_Employee', 
        'Submission_Conversion_Rate', 'Page_Depth'
    ]
    
    # ✅ Safely fill missing values before scaling
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    # Scale numerical features
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    # Handle categorical encoding
    # Low and medium cardinality columns (one-hot encoding)
    low_medium_cols = ['Deal Type', 'ICP Fit Level', 'Deal owner', 'Deal source attribution 2']
    
    # Create dummy columns for each categorical feature
    processed_df = df.copy()
    
    for col in low_medium_cols:
        # Create dummies for this column
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dummy_na=True)
        
        # Add missing columns that were in the training data
        for dummy_col in [c for c in encoding_info['one_hot_columns'] if c.startswith(col + '_')]:
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0
        
        # Keep only the columns that were in the training data
        dummies = dummies[[c for c in encoding_info['one_hot_columns'] if c.startswith(col + '_')]]
        
        # Add to the processed dataframe
        processed_df = pd.concat([processed_df, dummies], axis=1)
        
        # Drop the original column
        processed_df.drop(col, axis=1, inplace=True)
    
    # High cardinality columns (frequency encoding)
    high_cardinality_cols = ['State/Region', 'Primary Sub-Industry', 'Industry']
    
    for col in high_cardinality_cols:
        # Get the frequency map from training
        frequency_map = encoding_info['frequency_maps'][col]
        
        # Apply frequency encoding
        processed_df[f'{col}_freq'] = processed_df[col].map(
            lambda x: frequency_map.get(x, 1/(len(frequency_map)+1))
        )
        
        # Drop original column
        processed_df.drop(col, axis=1, inplace=True)
    
    return processed_df

