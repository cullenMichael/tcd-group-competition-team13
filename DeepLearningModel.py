import pandas as pd
import numpy as np
import pandas_profiling

def read_CSV_file(filename):
    return pd.read_csv(filename)


def main():
    # Read in the dataset to a pandas dataframe ,dtype={"Instance": int,"Crime Level in the City of Employement": int, "Year of Record": int,"Housing Situation": int,"Work Experience in Current Job [years]": int}
    dataset=read_CSV_file("tcd-ml-1920-group-income-train.csv")
    split_dataset = dataset.sample(frac=0.50,random_state=0)

    print("here!")
    # Use pandas profiling to view a gui based plot of the preprocessed data
    dataset.profile_report(title="initialData",check_recoded = False).to_file("initialData.html")




# run code
if __name__ == '__main__':
    main()
