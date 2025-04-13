import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_csv_from_github(file_name):
    base_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/refs/heads/main/"
    return pd.read_csv(base_url + file_name)


def prepare_features(df):
    current_year = pd.Timestamp.now().year

    # Drop columns with >=50% missing values and drop remaining rows with any missing values
    df = df[df.columns[df.isna().mean() < 0.5]].dropna()

    # Select relevant columns
    selected_columns = [
        'Annual Revenue', 'Number of Form Submissions', 'Web Technologies',
        'Number of times contacted', 'Time Zone', 'Primary Industry',
        'Number of Pageviews', 'Year Founded', 'Consolidated Industry',
        'Number of Employees', 'Number of Sessions', 'Country/Region', 'Industry'
    ]
    df_selected = df[selected_columns].copy()
    df_selected['Company_Age'] = current_year - df_selected['Year Founded']

    # Process top 20 Web Technologies
    all_techs = []
    for tech_list in df_selected['Web Technologies'].dropna():
        all_techs.extend([t.strip() for t in tech_list.split(';')])
    top_techs = [tech for tech, _ in pd.Series(all_techs).value_counts().head(20).items()]

    for tech in top_techs:
        df_selected[f'Has_{tech.replace(" ", "_")}'] = df_selected['Web Technologies'].apply(
            lambda x: 1 if pd.notna(x) and tech in x else 0
        )

    # One-hot encode other categorical columns
    categorical_columns = ['Time Zone', 'Primary Industry', 'Consolidated Industry', 'Country/Region', 'Industry']
    df_encoded = pd.get_dummies(df_selected, columns=categorical_columns, drop_first=True)

    # Scale numerical columns
    numerical_columns = [
        'Annual Revenue', 'Number of Form Submissions', 'Number of times contacted',
        'Number of Pageviews', 'Company_Age', 'Number of Employees', 'Number of Sessions'
    ]
    scaler = MinMaxScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

    # Prepare final feature set
    feature_columns = numerical_columns + [col for col in df_encoded.columns 
        if col.startswith('Has_') or any(col.startswith(prefix) for prefix in [
            'Time Zone_', 'Consolidated Industry_', 'Country/Region_', 'Industry_', 'Primary Industry_'])]

    X_cluster = df_encoded[feature_columns]
    return X_cluster, scaler, df_encoded
