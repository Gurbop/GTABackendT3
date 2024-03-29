import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle  # Use pickle for serialization
import numpy as np
class datamodel:
    def __init__(self, model_path="clashroyale.pkl", data_path="8V280L8VQ-clash-royale-da.csv"):
        self.model_path = model_path
        self.data_path = data_path
        # Attempt to load the model; if not found, initialize a new LogisticRegression model
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
                # Ensure the loaded object is an instance of a logistic regression model
                if not isinstance(self.model, LogisticRegression):
                    raise ValueError("Loaded model is not a LogisticRegression instance.")
        except (FileNotFoundError, ValueError) as e:
            print(f"Model load error: {e}. Initializing a new model.")
            self.model = LogisticRegression()
        self.data = pd.read_csv(data_path)  # Load data
    def display_data_head(self):
        print(self.data.head())
    def preprocess_data(self):
        self.X = self.data[['my_trophies', 'opponent_trophies', 'my_deck_elixir', 'op_deck_elixir']]
        self.Y = self.data['my_result']
    def train_model(self):
        self.preprocess_data()  # Ensure data is preprocessed
        self.model.fit(self.X, self.Y)
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir, op_deck_elixir):
        # Check if model has been trained or loaded correctly
        if isinstance(self.model, LogisticRegression):
            prediction = self.model.predict(np.array([[my_trophies, opponent_trophies, my_deck_elixir, op_deck_elixir]]))[0]
            return prediction
        else:
            raise TypeError("The model instance is not correctly initialized.")
    def add_data(self, new_data):
        self.data = self.data.append(new_data, ignore_index=True)
    def get_data(self):
        return self.data
    def update_data(self, index, updated_data):
        self.data.loc[index] = updated_data
    def delete_data(self, index):
        self.data.drop(index, inplace=True)
    def save_model(self):  # Explicitly save the model
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)
# Example usage
model = datamodel()  # Initialize the model and load data
model.train_model()  # Train the model before prediction
print(model.predict(111, 1, 5, 2))  # Example prediction
model.save_model()  # Save the model after training