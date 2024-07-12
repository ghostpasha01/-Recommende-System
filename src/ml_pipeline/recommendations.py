import numpy as np

def get_recommendations(model,user,items,user_to_product_interaction_matrix,user2index_map,product_to_feature_interaction_matrix):
    
    # getting the userindex
        
        userindex = user2index_map.get(user, None)
        
        if userindex == None:
            return None
        
        users = userindex
        
        # products already bought
        
        known_positives = items[user_to_product_interaction_matrix.tocsr()[userindex].indices]
        print('User index =',users)
        
        # scores from model prediction
        scores = model.predict(user_ids = users, item_ids = np.arange(user_to_product_interaction_matrix.shape[1]),item_features=product_to_feature_interaction_matrix)
        
        # top items
        
        top_items = items[np.argsort(-scores)]
        
        # printing out the result
        print("User %s" % user)
        print("     Known positives:") # already bought items
        
        for x in known_positives[:10]:
            print("                  %s" % x)
            
            
        print("     Recommended:") # items that are reccomeneded to a particular user
        
        for x in top_items[:10]:
            print("                  %s" % x)
    