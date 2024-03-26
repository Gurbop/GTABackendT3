import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
class datamodel:
    
    def __init__(self): # Loading in the model file(this includes all weights after training)
        self.model = joblib.load("clashroyale.pkl")
    def readfile(self):
        self.data=pd.read_csv("8V280L8VQ-clash-royale-da.csv")
        print(self.data.head())
    def preprocessing(self):# this is loading in the file stuff into X and Y
        self.X=self.data[['my_trophies','opponent_trophies','my_deck_elixir','op_deck_elixir']]
        self.Y=self.data['my_result']
    def train(self): # this is actually trianing the model
        self.model=LogisticRegression()
        self.model.fit(self.X,self.Y)
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir): # this is prediciton function
        return self.model.predict([[my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir]])[0]
    
    def exportmodel(self): #after trianing we have to export the file again
        joblib.dump(self.model,"clashroyale.pkl")
    
    def create(self, new_data):
        self.data = self.data.append(new_data, ignore_index=True)
        
    def read(self):
        return self.data
    
    def update(self, index, updated_data):
        self.data.loc[index] = updated_data
        
    def delete(self, index):
        self.data.drop(index, inplace=True)
    
    
Model=datamodel()# sample class thing

print(Model.predict(1,1,3.6,2.5))# sample Prediction