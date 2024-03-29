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
    def __init__(self):
        #loading the model file in
        #the pkl file needs to be loaded before we can do anything with it
        self.model = joblib.load("clashroyale.pkl")
    def readfile(self):
        # makes the csv file into data
        self.data=pd.read_csv("8V280L8VQ-clash-royale-da.csv")
        print(self.data.head())
    def create(self, new_data):
        # adding the new data to a list
        self.data = self.data.append(new_data, ignore_index=True)
    def exportmodel(self):
        #exporting the file after the training is done
        joblib.dump(self.model,"clashroyale.pkl")
    def read(self):
        # reading the data
        return self.data
    def update(self, index, updated_data):
        # updating data
        self.data.loc[index] = updated_data
    def delete(self, index):
        # deleting data
        self.data.drop(index, inplace=True)
Model=datamodel() # setting Model to the python file