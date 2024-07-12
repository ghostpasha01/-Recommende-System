import pandas as pd # pandas for data manipulation
import numpy as np # numpy for sure
from lightfm import LightFM # model
from lightfm.evaluation import auc_score
import time

# function to build a hybrid model with loss functions
def hybrid_model(loss,interaction_train,product_interaction):
    if loss=='warp': # loss function = WARP 
        model_with_features = LightFM(loss = "warp")
        start = time.time()

        model_with_features.fit_partial(interaction_train, 
                user_features=None, 
                item_features=product_interaction, 
                sample_weight=None, 
                epochs=1, 
                num_threads=4,
                verbose=False)
    
        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        return (model_with_features)


    elif loss=='logistic': # loss function = logistic
        model_with_features = LightFM(loss = "logistic",no_components=30)
        start = time.time()

        model_with_features.fit_partial(interaction_train,
            user_features=None, 
            item_features=product_interaction, 
            sample_weight=None, 
            epochs=10, 
            num_threads=20,
            verbose=False)

        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        return (model_with_features)

    elif loss=='bpr': # loss function = BPR

        model_with_features = LightFM(loss = "bpr")
        start = time.time()

        model_with_features.fit_partial(interaction_train,
            user_features=None, 
            item_features=product_interaction, 
            sample_weight=None, 
            epochs=1, 
            num_threads=4,
            verbose=False)
        end = time.time()
        print("time taken = {0:.{1}f} seconds".format(end - start, 2))
        return (model_with_features)

    else:
        print("none")



# function to evaluate the model with AUC score
def evaluate_model(model, interaction_test,interaction_train,product_interaction):

    start = time.time()

    auc_with_features = auc_score(model = model,  
                        test_interactions = interaction_test,
                        train_interactions = interaction_train, 
                        item_features = product_interaction,
                        num_threads = 4, check_intersections=False)

    end = time.time()

    print("time taken for AUC score= {0:.{1}f} seconds".format(end - start, 2))

    return("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))
