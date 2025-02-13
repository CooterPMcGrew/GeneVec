import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
import json

class DataClassifier:
    def __init__(self, file_path, target_column=None):
        self.file_path = file_path
        self.target_column = target_column
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoder = LabelEncoder()
    
    def load_data(self):
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    def preprocess_data(self):
        if self.target_column not in self.data.columns:
            raise ValueError("Target column not found in dataset.")
        
        self.data = self.data.dropna()
        X = self.data.drop(columns=[self.target_column])
        y = self.encoder.fit_transform(self.data[self.target_column])
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def classify_and_label(self, X_test):
        predictions = self.model.predict(X_test)
        labels = self.encoder.inverse_transform(predictions)
        return labels
    
    def run(self):
        self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.train_model(X_train, y_train)
        labels = self.classify_and_label(X_test)
        return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify and label data from a given file.')
    parser.add_argument('file_path', type=str, help='Path to the input data file (CSV or JSON).')
    parser.add_argument('target_column', type=str, help='Target column name for classification.')
    args = parser.parse_args()
    
    classifier = DataClassifier(args.file_path, args.target_column)
    output_labels = classifier.run()
    print("Classified Labels:", output_labels)
