import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
class datamodel:
    
    def __init__(self): # Loading in the model file(this includes all weights after training)
        self.model = joblib.load("clashroyale.pkl")
    def readfile(self):
        self.data=pd.read_csv("8V280L8VQ-clash-royale-da.csv")
        print(self.data.head())
        # loads the stuff in the file into X and Y
    def preprocessing(self):
        self.X=self.data[['my_trophies','opponent_trophies','my_deck_elixir','op_deck_elixir']]
        self.Y=self.data['my_result']
        # trains the model
    def train(self):
        self.model=LogisticRegression()
        self.model.fit(self.X,self.Y)
    def predict(self, my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir): # this is prediciton function
        return self.model.predict([[my_trophies, opponent_trophies, my_deck_elixir,op_deck_elixir]])[0]
    
    def exportmodel(self): 
        #after trianing we have to export the file again
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
