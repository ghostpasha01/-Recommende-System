import numpy as np
from scipy.sparse import coo_matrix # for constructing sparse matrix

def train_test_merge(training_data, testing_data):
    
    # initialising train dict
    train_dict = {}
    for row, col, data in zip(training_data.row, training_data.col, training_data.data):
        train_dict[(row, col)] = data
        
    # replacing with the test set
    
    for row, col, data in zip(testing_data.row, testing_data.col, testing_data.data):
        train_dict[(row, col)] = max(data, train_dict.get((row, col), 0))
        
    
    # converting to the row
    row_list = []
    col_list = []
    data_list = []
    for row, col in train_dict:
        row_list.append(row)
        col_list.append(col)
        data_list.append(train_dict[(row, col)])
        
    # converting to np array
    
    row_list = np.array(row_list)
    col_list = np.array(col_list)
    data_list = np.array(data_list)
    
    return coo_matrix((data_list, (row_list, col_list)), shape = (training_data.shape[0], training_data.shape[1]))