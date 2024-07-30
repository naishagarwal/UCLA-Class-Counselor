import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import numpy as np

# create df and lists of all scraped data 
df = pd.read_csv('cleaned_ucla_class_info_F_24.csv')
profs = df['class_professor'].to_list()
classes = df['class_name'].to_list()

# clean prof and class info to make it searchable
clean_profs = []
clean_classes = []
[clean_profs.append(item.replace(' ', '%20')) for item in profs]
[clean_classes.append(item.split('(')[-1].split(' -')[0].replace(')', '').replace(' ', '-').lower()) for item in classes]

def make_request(url):
    # scrape prof page for corresponding class rating
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, 'lxml')

    return soup

def get_full_names():
    # populate full names of profs to look up in bruin walk
    full_prof_names = []
    for prof in clean_profs:
        if prof == 'The%20Staff' or prof == 'TA':
            full_prof_names.append('None')
            continue
        course_url = 'https://bruinwalk.com/search/?q={}'.format(prof)
        soup = make_request(url=course_url)
        # try to query bruin walk for prof abbreviated name
        try:
            full_prof_names.append(soup.find('a', {'class': 'professor-name flex-item flex-middle'}).text.replace(' ', '-').lower())
        except:
            full_prof_names.append('None')
    
    return full_prof_names

def create_final_df(full_prof_names, clean_classes):
    # create df of data and write results to file
    final_df = pd.DataFrame(
        {'full_prof_name': full_prof_names,
        'class_name': clean_classes,
        })
    
    final_df.to_csv('prof_and_class_F_24.csv', index=False)
    
    return final_df

def get_rating():
    # make request for each prof+class combo and return rating
    df = pd.read_csv('prof_and_class_F_24.csv')
    profs = df['full_prof_name'].to_list()
    classes = df['class_name'].to_list()
    ratings = []
    urls = []
    for i, prof in enumerate(profs):
        # check for nan values and skip if nan
        if type(prof) is float:
            ratings.append('N/A')
            urls.append('N/A')
            continue

        url = 'https://bruinwalk.com/professors/{}/{}/'.format(prof, classes[i])
        # make request to bruin walk, append N/A if no page found
        soup = make_request(url)
        # try finding overall rating of specific prof+class combo
        try:
            rating = soup.find('div', {'class': 'overall-score'}).text
            ratings.append(rating.strip())
            urls.append(url)
        except:
            # try finding N/A rating when no ratings have been given
            try:
                rating = soup.find('div', {'class': 'overall-score na'}).text
                ratings.append(rating.strip())
                urls.append(url)
            except:
                # append prof website instead of prof+class website since it doesn't exist
                print('Bruin Walk page does not exist')
                ratings.append('N/A')
                urls.append(url.rsplit('/', 2)[0])
    
    return ratings, urls

def combine_df(ratings, urls):
    df = pd.read_csv('cleaned_ucla_class_info_F_24.csv')
    df['prof_rating'] = ratings
    df['bruin_walk_url'] = urls
    df.to_csv('prof_ratings_and_class_data_F_24.csv', index=False)

    # create dict for each record
    d = df.to_dict('records')

    # write results to file
    with open('prof_ratings_and_class_data_F_24.json', 'w') as fp:
        json.dump(d, fp)
        
### run these to get list of prof names and classes for each semester
#full_prof_names = get_full_names()
#create_final_df(full_prof_names, clean_classes)

### run this to get ratings for all profs and add to existing scraped data from registrar
#ratings, urls = get_rating()
#combine_df(ratings, urls)