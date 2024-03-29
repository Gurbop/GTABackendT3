import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

class datamodel:
    
    def __init__(self, model_path="clashroyale.pkl", data_path="8V280L8VQ-clash-royale-da.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = joblib.load(model_path)  # Load model
        self.data = pd.read_csv(data_path)  # Load data
        
    def display_data_head(self):
        print(self.data.head())
        
    def preprocess_data(self):
        self.X = self.data[['my_trophies', 'opponent_trophies', 'my_deck_elixir', 'op_deck_elixir']]
        self.Y = self.data['my_result']
    def train_model(self):
        self.preprocess_data()  # Ensure data is preprocessed
        self.model = LogisticRegression()
        self.model.fit(self.X, self.Y)
        
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir, op_deck_elixir):
        prediction = self.model.predict([[my_trophies, opponent_trophies, my_deck_elixir, op_deck_elixir]])[0]
        return prediction