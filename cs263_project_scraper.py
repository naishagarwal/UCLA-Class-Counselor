from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

### ensure chromedriver and chrome have comptaible versions and are installed in path ###

def scrape_department_codes():
    # get url and response from site
    departments_url = 'https://registrar.ucla.edu/faculty-staff/courses-and-programs/department-and-subject-area-codes'
    response = requests.get(departments_url)
    soup = BeautifulSoup(response.text, 'lxml')

    # extract table headers
    table = soup.find('table')
    table_headers_html = table.find_all('th')
    table_headers = []
    [table_headers.append(header.text) for header in table_headers_html]

    # extract table contents
    table_contents_html = table.find_all('td')
    table_contents = []
    [table_contents.append(row_item.text) for row_item in table_contents_html]

    # convert list into rows after every 6th item, convert table headers and contents into pandas dataframe
    content_length = len(table_contents)
    num_rows = content_length / 6
    abbreviation_df = pd.DataFrame(np.array(table_contents).reshape(int(num_rows),len(table_headers)), columns = table_headers)

    return abbreviation_df

# call function to scrape department codes
abbreviation_df = scrape_department_codes()

# convert department codes to format for URL input
#abbreviation_df['Abbreviation'].replace(' ', '+')
abbreviation_df['Abbreviation'] = abbreviation_df['Abbreviation'].replace(' ', '+', regex=True)

# write df to csv
abbreviation_df.to_csv('department_codes.csv', index=False)
print(abbreviation_df.head(5))