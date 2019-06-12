from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import helpers
import random
import pandas as pd

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
    if business_id:
        # content based fitering
        vergelijkbarebedrijven= helpers.cbf(business_id, city, n)
        return helpers.returndict(list(vergelijkbarebedrijven.index), city)
        
    # als user inlogt
    if user_id:
        userdata = helpers.location(user_id)
        stad = list(userdata[:1].index)[0]
        aantal_reviews = userdata.sum()
        
        if aantal_reviews < 5:
            # random op basis steden
            populairebedrijven = helpers.populair(stad, n)
            return helpers.returndict(list(populairebedrijven.index), stad)
        else:
            # cf op basis steden
            cflist = helpers.cf(user_id, stad, n)
            return helpers.returndict(list(cflist.index), stad)


    # bij geen andere mogelijkheid return random
    if not city:
        city = random.choice(CITIES)    
    return random.sample(BUSINESSES[city], n)