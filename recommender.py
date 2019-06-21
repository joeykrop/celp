from helpers import *
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

from heapq import nlargest

def training_test():
    df = json_to_df_stars()
    df_training, df_test = split_data(df, 0.5)
    return(df_training, df_test)

def predictions_item_based():
    df_training, df_test = training_test()
    
    df_utility_item = pivot_ratings(df_training)
    df_similarity_ratings = create_similarity_matrix_cosine(df_utility_item)
    
    return predict_ratings(df_similarity_ratings, df_utility_item, df_test).dropna()

def predictions_content_based():
    df_training, df_test = training_test()

    df_utility_ratings = pivot_ratings(df_training)
    df_categories = extract_categories()
    df_utility_content = pivot_genres(df_categories)
    df_similarity_content = create_similarity_matrix_categories(df_utility_content)
    
    return predict_ratings(df_similarity_content, df_utility_ratings, df_test).dropna()

def predictions_hybrid_based():
    item_prediction = predictions_item_based()
    content_prediction = predictions_content_based()
    combined = item_prediction
    combined['predict_content'] = content_prediction['predicted rating']
    combined['predicted rating'] = combined[['predicted rating', 'predict_content']].mean(axis=1)
    
    return item_prediction.drop(columns=['predict_content'])
    
def business_details(options):

    business_details = []
    for list in BUSINESSES.values():
        for dict in list:
            if dict['business_id'] in options:
                business_details.append(dict)
    return business_details


def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """

    if user_id == None:
        low_review_count = []
        for business in BUSINESSES[random.choice(CITIES)]:
            # only return businesses with less than 29 review count (threshold before averaging out)
            if business["review_count"] < 29:
                low_review_count.append(business)
        return random.sample(low_review_count, n)

    predictions = predictions_hybrid_based()
    user_prediction = predictions.loc[predictions['user_id'] == user_id]

    # keep the predictions that have a rating of 3.6 (average for dataset) and higher
    user_prediction = user_prediction.loc[user_prediction['predicted rating'] >= 3.6]
    business_predictions = user_prediction['business_id'].tolist()

    if len(business_predictions) < n:
        low_review_count = []
        for business in BUSINESSES[random.choice(CITIES)]:
            # only return businesses with less than 29 review count (threshold before averaging out)
            if business["review_count"] < 29:
                low_review_count.append(business)
        details = business_details(business_predictions) + random.sample(low_review_count, n - len(business_predictions))

    if len(business_predictions) >= n:
        rec_dict =  user_prediction.set_index('business_id')['predicted rating'].to_dict()
        best_business_predictions = nlargest(10, rec_dict, key=rec_dict.get)
        details = business_details(best_business_predictions)
        
    return details