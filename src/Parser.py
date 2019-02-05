# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:02:57 2019

@author: jamie.ross
"""

import pandas as pd

import os

#os.chdir('C:/Users/Jamie.Ross/Documents/Beer/')
#os.chdir('P:/temp/Jamie/Ad Hoc/Python/')


class Parser(object):
    
    def __init__(self, 
                 txt):
        
        with open(txt, 'r') as fls:
            self.txt = fls.read()
        self.parse(txt.replace('.txt', ''))
        
    def parse(self, number):

        #Identifier
        self.uid        = number
        self.name       = self.get_name()

        #Ratings        
        self.ratings    = self.get_ratings(number)
        
        #Ingredients
        self.ingredient = self.get_ingredients()
        
        #Descriptors
        self.brewer     = self.get_brewer()
        self.batch_size = self.get_batch_size() 
        self.boil_size  = self.get_boil_size() 
        self.colour     = self.get_colour() 
        self.bitter     = self.get_bitter()
        self.est_og     = self.get_est_og()
        self.est_fg     = self.get_est_fg()
        self.abv        = self.get_abv()
        self.style      = self.get_style() 
        self.boil_time  = self.get_boil_time() 
        self.mash_prof  = self.get_mash_prof()
        self.ferment    = self.get_fermentation()
        self.taste_rat  = self.get_taste_rating()
        
    #Ratings
    def get_ratings(self, number):
        txt = self.pattern('Ratings</h3>', '<h3>Become a Member') 
        txt = txt.split('comment')

        def sub_parse(txt, starte, ende):
            bege = txt.find(starte) + len(starte)
            ende = txt.find(ende, bege)
            txt  = txt[bege:ende]
            return txt.strip()
            
        if len(txt) < 2:
            return None
        
        ratings = []
        for ii in range(1, len(txt)):
            
            rating  = sub_parse(txt[ii], '.net/images/', '.png\\')
            user    = sub_parse(txt[ii], 'https://beersmithrecipes.com/viewuser/', '\\')    
            comment = sub_parse(txt[ii], '</span>\\n<p>', '</p>\\n')    
        
            ratings.append({'Rating'           : rating.replace('star', ''),
                            'User'             : user,
                            'Words'            : len(comment.split(' ')),
                            'Char'             : len(comment)})
        ratings           = pd.DataFrame.from_dict(ratings)
        ratings['Recipe'] = number.replace('recipes/', '')
        return ratings
        
    #Ingredients
    def get_ingredients(self):
        def cleaner(txt):
            txt_use = txt
            txt_use = txt_use.replace('</td>', '')
            txt_use = txt_use.replace('</tr>', '')
            txt_use = txt_use.replace('<tr>', '')
            txt_use = txt_use.replace('\\n', '')
            txt_use = txt_use.replace('<tr class=\\', '')
            txt_use = txt_use.replace('alt\\', '')
            txt_use = txt_use.replace('>', '')
            txt_use = txt_use.replace("'", '')
            return txt_use
        
        txt = self.pattern('>Ingredients<', '</table>') 
        txt = txt.split('<td>')        
        txt = txt[1:]
        txt = [cleaner(x) for x in txt]
    
        ingred = []
        for ii in range(int(len(txt) / 4)):
            
            ingred.append(pd.DataFrame({'Amount'       : [txt[4 * ii + 0]],
                                        'Name'         : [txt[4 * ii + 1]],
                                        'Type'         : [txt[4 * ii + 2]],
                                        'Order'        : [txt[4 * ii + 3]]}))
        return pd.concat(ingred)                            
            
    #Descriptors
    def get_name(self):
        txt = self.pattern('<title>', '</title>') 
        return txt.replace(' - Recipe - BeerSmith Cloud', '')
    
    def get_brewer(self):
        return self.pattern('Brewer: </b>', '</td>') 
    
    def get_batch_size(self):
        return self.pattern('Batch Size: </b>', '</td>') 

    def get_boil_size(self):
        return self.pattern('Boil Size: </b>', '</td>') 

    def get_colour(self):
        return self.pattern('Color:</b>', '</td>') 

    def get_bitter(self):
        return self.pattern('Bitterness:</b>', '</td>') 

    def get_est_og(self):
        return self.pattern('Est OG:</b>', '</td>') 

    def get_est_fg(self):
        return self.pattern('Est FG:</b>', '</td>') 

    def get_abv(self):
        return self.pattern('ABV:</b>', '</td>') 

    def get_style(self):
        return self.pattern('Style:</b>', '</td>') 

    def get_boil_time(self):
        return self.pattern('Boil Time:</b>', '</td>') 

    def get_mash_prof(self):
        return self.pattern('Mash Profile:</b>', '</td>') 

    def get_fermentation(self):
        return self.pattern('Fermentation:</b>', '</td>') 

    def get_taste_rating(self):
        return self.pattern('Taste Rating:</b>', '</td>') 
    
    #Utitlities
    def pattern(self, bege, ende):
        bege = self.txt.find(bege) + len(bege)
        ende = self.txt.find(ende, bege)
        txt  = self.txt[bege:ende]
        return txt.strip()
        
if __name__ == '__main__':
    fls    = os.listdir('recipes/')
    receps = []
    
    for fl in fls:
        beer = Parser('recipes/' + fl)
        print(fl)
        if beer.ratings is not None:
            receps.append(beer)
