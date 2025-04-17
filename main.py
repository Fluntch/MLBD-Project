from src import data_cleaning
from src import features
from src import model
import seaborn as sns
from src.config import *

def main():
    sns.set_palette(["#D3D3D3"])

    if PREPROCESS:
        print("=======Starting data cleaning=======")
        data_cleaning.clean()
        print("=======Finished data cleaning=======")

    if EXTRACT:
        print("=======Starting to extract features======")
        features.extract()
        print("=======Finished extracting features=======")

    if MODEL:
        print("=======Starting to model data======")

        print("=======Finished data modelling=======")


if __name__ == '__main__':
    main()

