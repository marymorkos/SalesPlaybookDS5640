"""
preprocessing.py - Preprocessing module for clustering data

This module contains functions to clean, transform, and prepare data for clustering.
"""

import pandas as pd
import numpy as np
import datetime
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def remove_high_missing_columns(df, threshold=0.5):
    """
    Remove columns with missing values above the specified threshold.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    threshold : float, default=0.5
        Maximum percentage of missing values allowed
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with columns removed that exceed the missing value threshold
    """
    # Calculate the percentage of missing values in each column
    missing_percentage = df.isna().mean()
    
    # Filter columns that have less than threshold missing values
    columns_to_keep = missing_percentage[missing_percentage < threshold].index.tolist()
    
    # Create a new DataFrame with only these columns
    df_cleaned = df[columns_to_keep]
    
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Cleaned DataFrame shape: {df_cleaned.shape}")
    print(f"Removed {df.shape[1] - df_cleaned.shape[1]} columns with {threshold*100}% or more missing values")
    
    return df_cleaned

def drop_missing_rows(df):
    """
    Drop all rows that contain any missing values.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with rows removed that contain any missing values
    """
    df_cleaned = df.dropna()
    print(f"Shape after dropping rows: {df_cleaned.shape}")
    return df_cleaned

def select_columns(df, columns):
    """
    Select specific columns from the DataFrame.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    columns : list
        List of column names to select
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with only the selected columns
    """
    return df[columns].copy()

def create_company_age(df, year_col='Year Founded'):
    """
    Create a company age feature based on the year founded.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    year_col : str, default='Year Founded'
        Name of the column containing the founding year
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with the additional Company_Age column
    """
    df = df.copy()
    current_year = datetime.datetime.now().year
    df['Company_Age'] = current_year - df[year_col]
    return df

def create_technology_features(df, tech_column='Web Technologies', top_n=20):
    """
    Create dummy variables for the most common technologies.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    tech_column : str, default='Web Technologies'
        Name of the column containing the technology information
    top_n : int, default=20
        Number of top technologies to create features for
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with additional binary technology features
    """
    df = df.copy()
    
    # First, create a list of all technologies by splitting the strings
    all_technologies = []
    for tech_list in df[tech_column].dropna():
        techs = [t.strip() for t in tech_list.split(';')]
        all_technologies.extend(techs)
    
    # Get the top N most common technologies
    top_techs = Counter(all_technologies).most_common(top_n)
    print(f"Top {top_n} technologies:")
    for tech, count in top_techs:
        print(f"{tech}: {count}")
    
    # Create dummy variables for the top technologies
    for tech, _ in top_techs:
        df[f'Has_{tech.replace(" ", "_")}'] = df[tech_column].apply(
            lambda x: 1 if pd.notna(x) and tech in x else 0
        )
    
    return df

def one_hot_encode(df, categorical_columns):
    """
    Perform one-hot encoding on categorical variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    categorical_columns : list
        List of categorical column names to encode
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with one-hot encoded categorical variables
    """
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def scale_numerical_features(df, numerical_columns):
    """
    Scale numerical features to the 0-1 range using MinMaxScaler.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process
    numerical_columns : list
        List of numerical column names to scale
        
    Returns:
    --------
    tuple
        (scaled DataFrame, fitted scaler)
    """
    # Create a copy to avoid warnings
    df_scaled = df.copy()
    
    # Apply MinMaxScaler to numerical columns only
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns].fillna(0))
    
    # Print min and max values to verify scaling
    print("Min and max values for numerical features after scaling:")
    for col in numerical_columns:
        print(f"{col}: Min = {df_scaled[col].min():.4f}, Max = {df_scaled[col].max():.4f}")
    
    print(f"Shape after scaling: {df_scaled.shape}")
    
    return df_scaled, scaler

def get_clustering_features(df, numerical_columns):
    """
    Create the feature set for clustering by selecting appropriate columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The processed and scaled DataFrame
    numerical_columns : list
        List of numerical column names
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing only the features to use for clustering
    """
    columns_to_keep = numerical_columns.copy()
    
    # Add all the dummy columns created from one-hot encoding and technology flags
    for col in df.columns:
        if col not in numerical_columns and col != 'Web Technologies' and not col.endswith('Cluster'):
            if (col.startswith('Time Zone_') or  
                col.startswith('Consolidated Industry_') or 
                col.startswith('Country/Region_') or 
                col.startswith('Industry_') or
                col.startswith('Has_')):
                columns_to_keep.append(col)
    
    return df[columns_to_keep]

def full_preprocessing_pipeline(df):
    """
    Run the full preprocessing pipeline on the input DataFrame.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The raw input DataFrame
        
    Returns:
    --------
    tuple
        (fully processed DataFrame for clustering, fitted scaler, feature DataFrame)
    """
    # Define the columns we want to select
    selected_columns = [
        'Annual Revenue', 
        'Number of Form Submissions', 
        'Web Technologies', 
        'Number of times contacted',  
        'Time Zone', 
        'Primary Industry', 
        'Number of Pageviews', 
        'Year Founded',  
        'Consolidated Industry', 
        'Number of Employees', 
        'Number of Sessions', 
        'Country/Region', 
        'Industry'
    ]
    
    # Define categorical columns
    categorical_columns = [
        'Time Zone', 
        'Primary Industry', 
        'Consolidated Industry', 
        'Country/Region', 
        'Industry'
    ]
    
    # Define numerical columns
    numerical_columns = [
        'Annual Revenue', 
        'Number of Form Submissions', 
        'Number of times contacted',
        'Number of Pageviews', 
        'Company_Age',  # Note: This is created in the pipeline
        'Number of Employees', 
        'Number of Sessions'
    ]
    
    # Step 1: Remove columns with high missing values
    df_cleaned = remove_high_missing_columns(df)
    
    # Step 2: Drop rows with any missing values
    df_cleaned = drop_missing_rows(df_cleaned)
    
    # Step 3: Select relevant columns
    df_selected = select_columns(df_cleaned, selected_columns)
    
    # Step 4: Create company age feature
    df_selected = create_company_age(df_selected)
    
    # Step 5: Create technology features (with top 20 technologies)
    df_selected = create_technology_features(df_selected, tech_column='Web Technologies', top_n=20)
    
    # Step 6: One-hot encode categorical variables
    df_encoded = one_hot_encode(df_selected, categorical_columns)
    
    # Step 7: Scale numerical features
    df_scaled, scaler = scale_numerical_features(df_encoded, numerical_columns)
    
    # Step 8: Get features for clustering
    X_cluster = get_clustering_features(df_scaled, numerical_columns)
    
    return df_scaled, scaler, X_cluster

if __name__ == "__main__":
    # Example usage
    print("This module contains preprocessing functions for clustering.")
    print("Import and use the functions in your main script.")
