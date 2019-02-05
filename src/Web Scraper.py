# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 12:40:19 2019

@author: jamie.ross
"""

from requests import get
from requests.exceptions import RequestException
from contextlib import closing

import time
import os

#os.chdir('C:/Users/Jamie.Ross/Documents/Beer/')

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)
    
def main():
    """
    Loop over webpages, and save if interesting
    """
    fail_ind = 0    
    for ii in range(10000, 15000): #Pause for 
        url = 'https://beersmithrecipes.com/viewrecipe/' + str(ii)
        rec = simple_get(url)
        rec = str(rec)
        
        if ii % 10 == 0:
            print(ii)
        
        if rec.find('Brewer: </b>') < 0:
            fail_ind += 1
            if fail_ind > 500:
                break
            continue
        else:
            fail_ind = 0
        
        time.sleep(1)        
        myfile = open('Recipes/' + str(ii) + '.txt', "w")  
        myfile.write(str(rec))
        myfile.close()    

if __name__ == '__main__':
    main()


