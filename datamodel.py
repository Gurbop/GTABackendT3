import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
class datamodel:
    def preprocessing(self):# this is loading in the file stuff into X and Y
        self.X=self.data[['my_trophies','opponent_trophies','my_deck_elixir','op_deck_elixir']]
        self.Y=self.data['my_result']
    def train(self): # this is actually trianing the model
        self.model=LogisticRegression()
        self.model.fit(self.X,self.Y)
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir): # this is prediciton function
        return self.model.predict([[my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir]])[0]