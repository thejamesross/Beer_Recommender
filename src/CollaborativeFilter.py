# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:43:29 2019

@author: jamie.ross
"""

ratings = [beer.ratings for beer in receps]
ratings = pd.concat(ratings)
ratings = ratings[[x in ['10', '20', '30', '40', '50'] for x in ratings.Rating]]

ratings['Char']   = pd.to_numeric(ratings['Char'])
ratings['Rating'] = pd.to_numeric(ratings['Rating'])
ratings['Words']  = pd.to_numeric(ratings['Words'])

#Tomorrow --

#Page Rank

#Collaborative filtering

#Clean Recipe data for model
    # Predictive Model
    #Optimize over inputs
    