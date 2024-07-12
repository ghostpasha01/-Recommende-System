import pandas as pd
import numpy as np


# Creating the list of unique users
def unique_users(data, column):   
    return np.sort(data[column].unique())

# Creating the list of unique produts
def unique_items(data, column):
    item_list = data[column].unique()
    return item_list

def features_to_add(customer, column1,column2,column3):
    customer1 = customer[column1]
    customer2 = customer[column2]
    customer3 = customer[column3]
    return pd.concat([customer1,customer3,customer2], ignore_index = True).unique()

# Create id mappings to convert user_id, item_id, and feature_id
def mapping(users, items, features):
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(users):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id
        
    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(items):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id
        
    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(features):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id
        
        
    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping
