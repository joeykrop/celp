from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from pandas.io.json import json_normalize

import sklearn.metrics.pairwise as pw 
import numpy as np
import pandas as pd
import random

## General helper functions ##

def split_data(data, d):
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]

def json_to_df_stars():
    df = pd.DataFrame()

    # add each city's DataFrame to the general DataFrame
    for city in CITIES:
        df = df.append(pd.DataFrame.from_dict(json_normalize(REVIEWS[city]), orient='columns'))
    
    # drop repeated user/business reviews and only save the latest one (since that one is most relevant)
    df = df.drop_duplicates(subset=["business_id", "user_id"], keep="last").reset_index()[["business_id", "stars", "user_id"]]
    return df

def mse(predicted_ratings):
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()

## Item based helper functions ##

def pivot_ratings(df):
    return df.pivot(values='stars', columns='user_id', index='business_id')

def create_similarity_matrix_cosine(matrix):
    mc_matrix = matrix - matrix.mean(axis = 0)
    return pd.DataFrame(pw.cosine_similarity(mc_matrix.fillna(0)), index = matrix.index, columns = matrix.index)

def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return np.nan
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return np.nan

def predict_ratings(similarity, utility, to_predict):
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id']), axis=1)
    return ratings_test_c

## Content based helper functions ##

def categories_dataframe():
    # Make a dataframe with the business_id's and their categories
    all_data = list()
    for businesses in BUSINESSES.values():
        for business in businesses:
            business_id = business['business_id']
            categories = business['categories']
    
    # add to the data collected so far
            all_data.append([business_id, categories])

    # create the DataFrame
    categories_df = pd.DataFrame(all_data, columns=['business_id', 'categories'])
    return categories_df

def extract_categories():
    """" extract the categories"""
    businesses = categories_dataframe()

    # replace nan with emptry string
    businesses = businesses.fillna('')
    categories_b = businesses.apply(lambda row: pd.Series([row['business_id']] + row['categories'].split(", ")), axis=1)
    stack_categories = categories_b.set_index(0).stack()
    df_stack_categories = stack_categories.to_frame()
    df_stack_categories['business_id'] = stack_categories.index.droplevel(1)
    df_stack_categories.columns = ['categories', 'business_id']
    return df_stack_categories.reset_index()[['business_id', 'categories']]

def pivot_genres(df):
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def create_similarity_matrix_categories(matrix):
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)