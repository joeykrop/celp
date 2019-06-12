from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import random
import pandas as pd
import sklearn.metrics.pairwise as pw
import numpy as np
import data
import math


def cf(userid, stad, n):
    """
    berekent het cijfer wat de gebruiker zou geven
    returnt de beste bedrijven naar aanleiding van het cijfer
    """
    # maak utilitymatrix van alle bedrijven
    ratings = pd.DataFrame.from_dict(REVIEWS[stad])
    utilitymatrix = ratings.pivot_table(values='stars', columns='user_id', index='business_id', aggfunc='mean')
    # mean centered ratings
    centered_utilitymatrix = utilitymatrix.sub(utilitymatrix.mean())
    similaritymatrix = pd.DataFrame(pw.cosine_similarity(centered_utilitymatrix.fillna(0)), index = centered_utilitymatrix.index, columns = centered_utilitymatrix.index)

    # bereken voor elk bedrijf wat de verwachte rating zal zijn voor user
    data=[]
    indexes=[]
    gereviewed = get_businesses(userid)
    for bedrijf in BUSINESSES[stad]:
        if bedrijf['business_id'] not in gereviewed:
            neighborhood = select_neighborhood(similaritymatrix, centered_utilitymatrix, userid, bedrijf['business_id'])
            data.append(weighted_mean(neighborhood, centered_utilitymatrix, userid))
            indexes.append(bedrijf['business_id'])
    predicted_ratings = pd.Series(index=indexes, data=data)
    gemiddelde_rating = utilitymatrix.mean()
    predicted_ratings = predicted_ratings + gemiddelde_rating[userid]

    # calculate the mse average rating wise
    # dit kan alleen als regel 27 weg is in de cf functie, want anders berekent hij geen rating voor het gereviewde bedrijf zelf
    # dit is gedaan omdat je deze aanbeveling niet wilt tussen de aanbevolen businesses
    # tomse = predicted_ratings.to_frame()
    # tomse['gem_rating'] = gemiddelde_rating[userid]
    # tomse["user_rating"] = np.nan
    # for bedrijf in tomse.index:
    #     for review in REVIEWS[stad]:
    #         if review['user_id'] == userid:
    #             if review['business_id'] == bedrijf:
    #                 tomse.at[bedrijf, 'user_rating'] = review['stars']
    # print(mse(tomse))

    # return de hoogste n bedrijven
    return predicted_ratings.sort_values(ascending=False)[:n]


def populair(stad, n):
    """
    returnt een Series van populairste bedrijven in de stad
    """
    data=[]
    indexes=[]
    # loop door alle bedrijven in de stad
    for bedrijf in BUSINESSES[stad]:
        # als ze boven de drempelwaarde zitten
        if bedrijf["stars"] > 3:
            # voeg de gegevens toe aan de datalists
            data.append(bedrijf['stars'] * bedrijf["review_count"])
            indexes.append(bedrijf["business_id"])

    # maak de gesorteerde series en return deze
    return pd.Series(data= data, index= indexes).sort_values(ascending=False)[:n]

def cbf(business_id, city, n):
    """
    bepaalt vergelijkbare bedrijven dmv content based filtering
    returnt de best vergelijkbare bedrijven en de stad
    """
    # haal de categories op van het bedrijf
    cats = data.get_business(city, business_id)['categories'].split(', ')
    indexes=[]
    overlapdata=[]

    # vergelijk cats met andere bedrijven dmv keyword overlap
    for bedrijf in BUSINESSES[city]:
        if business_id != bedrijf["business_id"]:
            catstemp = data.get_business(city, bedrijf["business_id"])['categories'].split(', ')
            counter = 0
            for cat in cats:
                if cat in catstemp:
                    counter += 1

            # calculate the similarity per business
            sim = counter / (len(cats) + len(catstemp) - counter)
            indexes.append(bedrijf['business_id'])
            overlapdata.append(sim)

    # return the best matching businesses
    return pd.Series(index=indexes, data=overlapdata).sort_values(ascending=False)[:n]


def returndict(lijst, stad):
    """
    returnt de lijst van bedrijven in het correcte format
    """
    recommended = []
    for bedrijf in lijst:
        gegevens = data.get_business(stad, bedrijf)
        recommended.append(gegevens)
    return recommended

    
def location(userid):
    """
    bepaalt de plaats waar de gebruiker 'woont'
    """
    # lijst aanmaken waar alle plaatsen van user inkomen
    plaatsen = []
    # per stad kijken of user in stad heeft gereviewt
    for stad in CITIES:
        for user in USERS[stad]:
            if user['user_id'] == userid:
                plaatsen.append(stad)
    # lijst met aantal reviews per plaats door user            
    counts = []
    for stad in plaatsen:
        count = 0
        for review in REVIEWS[stad]:
            if review['user_id'] == userid:
                count += 1
        counts.append(count)
    # series aanmaken waarin het aantal reviews per plaats staan
    reviews_plaats = pd.Series(index=plaatsen, data=counts).sort_values(ascending=False)
    return reviews_plaats


def select_neighborhood(similarity_matrix, utility_matrix, target_user, target_business):
    """
    selects all items with similarity > 0
    """
    # take the businesses where similarity is above 0, if it doesn't work it is an empty series
    try:
        bedrijven = similarity_matrix[similarity_matrix[target_business] > 0][target_business]
    except:
        return pd.Series()
    # get the business out which have not been reviewed yet
    try:
        eruit = utility_matrix[np.isnan(utility_matrix[target_user])].index
    except: 
        # else every business has been reviewed, but still drop the target_business
        return bedrijven
        # return bedrijven.drop(target_business)

    # try to get every not seen movie out of the neighborhood, if it is not in a try statement it will give errors
    for bedrijf in eruit:
        try:
            bedrijven = bedrijven.drop(bedrijf)
        except:
            pass
    return bedrijven


def weighted_mean(neighborhood, utility_matrix, user_id):
    """
    bepaalt het verachte cijfer dat de gebruiker zou geven
    """
    neighbor_ratings = pd.Series()
    # for every neighbor,
    for i in neighborhood.index:
        try:
            # calcute the product of the neighborhood and similarity
            neighbor_ratings.at[i] = utility_matrix[user_id][i] * neighborhood[i]
        except:
            # else it doesn't work and should return NaN
            neighbor_ratings.at[i] = np.nan

    # if neighborhood.values.sum is zero, it cannot devide
    try:
        return neighbor_ratings.sum() / neighborhood.values.sum()
    except:
        return np.nan


def get_businesses(user_id):
    """
    finds all the businesses a user has reviewed
    """
    businesslijst = set()

    # find every business that the user has reviewed.
    for city in CITIES:
        for i in REVIEWS[city]:
            if i["user_id"] == user_id:
                businesslijst.add(i["business_id"])
    return businesslijst


def mse(predicted_ratings):
    """
    calculates the mse of two rows in a dataframe
    """
    teller = 0
    noemer = 0
    for i in predicted_ratings.index:
        werkelijk = predicted_ratings.at[i, 'user_rating']
        verwacht = predicted_ratings.at[i, "gem_rating"]
        if np.isnan(werkelijk) == False and np.isnan(verwacht) == False:
            teller += (werkelijk - verwacht)**2
            noemer += 1
    return teller / noemer