import pandas as pd
import streamlit as st

def load_data(csv_path:str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


### Usage:
### from utils.data_loader import load_data
# df = load_data("data/application_train.csv")

              
    