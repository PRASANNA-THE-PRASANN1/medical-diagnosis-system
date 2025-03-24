import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_preprocess_data():
    # Load the Pima Indians Diabetes dataset from URL
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    
    # Replace zeros with NaN for columns where 0 is an invalid value, then fill with the column mean
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save the trained model and scaler in the models folder
    joblib.dump(rf_model, 'models/diabetes_rf_model.joblib')
    joblib.dump(scaler, 'models/diabetes_scaler.joblib')
    
    # Evaluate and print the test accuracy
    accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Test Accuracy: {accuracy:.4f}")