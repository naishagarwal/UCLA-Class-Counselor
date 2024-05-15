from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import numpy as np

# function to initialize selenium webdriver
def init_driver():
    options = webdriver.ChromeOptions()
    # enable headless option
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

# function to scrape the class content
def get_class_content(major, driver, year, semester):

    try:
        # wait for the shadow root to be available
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ucla-sa-soc-app'))
        )

        # find the element within the shadow root and click to expand class info
        shadow_root_script = 'return document.querySelector("#block-mainpagecontent > div > div > div > div > ucla-sa-soc-app").shadowRoot'
        shadow_root = driver.execute_script(shadow_root_script)
        element_within_shadow_root = shadow_root.find_element(By.CSS_SELECTOR, '#expandAll')
        element_within_shadow_root.click()

        # sleep to let the page fully populate
        time.sleep(6)

        # find all id's that start with major acronym
        shadow_class_info = shadow_root.find_elements(By.CSS_SELECTOR, '[id^={}]'.format(major.replace('+', '')))

        # create lists of data to keep from all shadow class info
        lines_to_keep = []
        for class_ in shadow_class_info:
            #with open('ucla_class_info.txt','a', newline='\n') as file:
            info_splits = class_.text.split('\n')
            for item in info_splits:
                if (('Select' in item and 'Lec' in item)
                     or ('am-' in item or 'pm-' in item) 
                     or ('TR' in item or 'MW' in item or item=='M' or item=='T' or item=='W' or item=='R' or item=='F' or item=='MWF')):
                    lines_to_keep.append(item)

        # store the items we need in a list
        final_lines = []
        for i, line in enumerate(lines_to_keep):
            if ('Select' in line and line.split('Select ')[-1] not in final_lines):
                # append class name once
                final_lines.append(line.split('Select ')[-1])
                # append class dates
                final_lines.append(lines_to_keep[i+1])
                # append class times
                final_lines.append(lines_to_keep[i+2])

        # write out semesters
        if semester == 'S':
            semester = 'spring'
        if semester == 'F':
            semester = 'fall'
        if semester == 'W':
            semester = 'winter'
            
        # define dataframe column headers, calculate dimensions of new np array for df, and append year and semester to df
        cols = ['class_name', 'class_dates', 'class_times']
        num_cols = len(cols)
        num_rows = int(len(final_lines)/num_cols)
        df = pd.DataFrame(np.array(final_lines).reshape(num_rows, num_cols), columns = cols)
        df['year'] = '20{}'.format(year)
        df['semester'] = semester

        return df, shadow_root

    except:
        print('get_class_content failed')
        df = ('', '')
        return df

# function to click next page
def click_next_page(page_count, driver, shadow_root):

    try:
        # wait for the shadow root to be available
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ucla-sa-soc-app'))
        )

        # look for next page button in shadow root, move to it, and click on it 
        button_within_shadow_root = shadow_root.find_element(By.CSS_SELECTOR, '#divPagination > div:nth-child(2) > ul > li:nth-child({}) > button'.format(str(page_count)))
        ActionChains(driver).move_to_element(button_within_shadow_root).click().perform()

        # give the page time to load after clicking
        time.sleep(6)

        return 'success'

    except:
        print('click_next_page failed')
        return 'fail'

# initialize page counts, years to be scraped (2020-2024), and semesters
page_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
years = [20, 21, 22, 23, 24]
semesters = ['F', 'W', 'S']

# read in majors to scrape and drop dupulicates
majors = pd.read_csv('majors_to_scrape.csv')
majors = majors['Abbreviation'].drop_duplicates().tolist()

# get info for each major for each year for each semester
for major in majors:
    for year in years:
        for semester in semesters:
            course_url = 'https://sa.ucla.edu/ro/Public/SOC/Results?t={}{}&sBy=subject&subj={}'.format(str(year), semester, major)

            # initialize driver and get url
            driver = init_driver()
            driver.get(course_url)

            # get initial df
            df = get_class_content(major, driver, year, semester)[0]
            shadow_root = get_class_content(major, driver, year, semester)[1]
            try:
                print(df.head())
                df.to_csv('ucla_class_info.csv', mode='a', header=False, index=False)
            except:
                print('df was empty')

            # iterate over all pages
            for page_count in page_counts:
                page_click = click_next_page(page_count, driver, shadow_root)
                if page_click == 'success':    
                    df = get_class_content(major, driver, year, semester)[0]
                    shadow_root = get_class_content(major, driver, year, semester)[1]
                    try:
                        print(df.head())
                        df.to_csv('ucla_class_info.csv', mode='a', header=False, index=False)
                    except:
                        print('df was empty')
                else:
                    print('no more page clicks')
                    break

            # close webdriver and continue to next semester
            driver.quit()
            continue
        driver.quit()
        continue
    driver.quit()
    continue