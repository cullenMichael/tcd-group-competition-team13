import pandas as pd
import numpy as np
import pandas_profiling

def read_CSV_file(filename):
    return pd.read_csv(filename)


def main():
    # Read in the dataset to a pandas dataframe 
    dataset=read_CSV_file("tcd-ml-1920-group-income-train.csv")

    print("here!")
    # Use pandas profiling to view a gui based plot of the preprocessed data
    dataset.profile_report(title="initialData",check_recoded = False).to_file("initialData.html")




# run code
if __name__ == '__main__':
    main()
