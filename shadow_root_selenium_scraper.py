from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import numpy as np

# url of the webpage you want to scrape
year = 23
major = 'COM+SCI'
course_url = 'https://sa.ucla.edu/ro/Public/SOC/Results?t={}S&sBy=subject&subj={}'.format(year, major)

# function to initialize selenium webdriver
def init_driver():
    options = webdriver.ChromeOptions()
    # Add any options you need, for example, to run headless
    # options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

# function to scrape the specified JS path
def scrape_path():
    driver = init_driver()
    driver.get(course_url)

    try:
        # wait for the shadow root to be available
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ucla-sa-soc-app'))
        )

        ### clicks "expand all classes"
        shadow_root_script = 'return document.querySelector("#block-mainpagecontent > div > div > div > div > ucla-sa-soc-app").shadowRoot'
        shadow_root = driver.execute_script(shadow_root_script)

        # find the element within the shadow root and click to expand class info
        element_within_shadow_root = shadow_root.find_element(By.CSS_SELECTOR, '#expandAll')
        element_within_shadow_root.click()

        # sleep to let the page fully populate
        time.sleep(4)

        # find all id's that start with major acronym
        major = 'COMSCI'
        shadow_class_info = shadow_root.find_elements(By.CSS_SELECTOR, '[id^={}]'.format(major))

        # initially create text file or overwrite old contents
        #with open('ucla_class_info.txt','w') as file:
        #    file.write('')

        # create lists of data to keep from all shadow class info
        lines_to_keep = []
        for class_ in shadow_class_info:
            #with open('ucla_class_info.txt','a', newline='\n') as file:
            info_splits = class_.text.split('\n')
            for item in info_splits:
                if ('Select' in item and 'Lec' in item) or ('am-' in item or 'pm-' in item) or ('TR' in item or 'MW' in item):
                    lines_to_keep.append(item)
                    #file.write(item + '\n')

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
            
        # define dataframe column headers, calculate dimensions of new np array for df
        cols = ['class_name', 'class_dates', 'class_times']
        num_cols = len(cols)
        num_rows = int(len(final_lines)/num_cols)
        df = pd.DataFrame(np.array(final_lines).reshape(num_rows, num_cols), columns = cols)

        print(df.head())

    finally:
        # close webdriver
        driver.quit()

# call primary function
scrape_path()