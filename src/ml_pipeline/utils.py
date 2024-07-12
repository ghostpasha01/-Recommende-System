import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix # for constructing sparse matrix


# Function to read excel file
def read_data(filepath,data):
    try:
        df = pd.read_excel(filepath,data)
    except Exception as e:
        print(e)
    else:
        return df


# Function to merge different data files 
def merge_dataset(df1, df2, left_on_param, right_on_param, join_type):
    try:
        final_df = pd.merge(df1, df2, left_on = left_on_param, right_on = right_on_param, how = join_type)
    except Exception as e:
        print(e)
    else:
        return final_df

# function to interaction matrix
def interactions(data, row, col, value, row_map, col_map):
    
    row = data[row].apply(lambda x: row_map[x]).values
    col = data[col].apply(lambda x: col_map[x]).values
    value = data[value].values
    
    return coo_matrix((value, (row, col)), shape = (len(row_map), len(col_map)))