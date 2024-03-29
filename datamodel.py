import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
class datamodel:
    def preprocessing(self):# this is loading in the file stuff into X and Y
        self.X=self.data[['my_trophies','opponent_trophies','my_deck_elixir','op_deck_elixir']]
        self.Y=self.data['my_result']
    def train(self): 
        #running the logisitcal regression function
        self.model=LogisticRegression()
        self.model.fit(self.X,self.Y)
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir):
        #predicting the result of victory or defeat based on your trophies and elixir and opponent elixir and trophies
        return self.model.predict([[my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir]])[0]