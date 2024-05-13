import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

df= pd.read_csv('department_codes.csv')

# drop duplicate abbreviations and convert to list
abb_list = df['Abbreviation'].drop_duplicates().tolist()

# check if there are classes available in Spring 2024 
year = 24
semester = 'S'

# initialize list
are_classes_list = []

# iterate through majors
for major in abb_list:
    # initialize chromedriver
    options = webdriver.ChromeOptions()
    # Add any options you need, for example, to run headless
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    course_url = 'https://sa.ucla.edu/ro/Public/SOC/Results?t={}{}&sBy=subject&subj={}'.format(year, semester, major)
    driver.get(course_url)
    try:
        # wait for the shadow root to be available
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ucla-sa-soc-app'))
        )

        ### finds "expand all classes"
        shadow_root_script = 'return document.querySelector("#block-mainpagecontent > div > div > div > div > ucla-sa-soc-app").shadowRoot'
        shadow_root = driver.execute_script(shadow_root_script)

        # find the element within the shadow root to confirm there are classes present and denote in list if successful
        element_within_shadow_root = shadow_root.find_element(By.CSS_SELECTOR, '#expandAll')
        are_classes_list.append(1)
    
    except:
        # no classes appeared for major 
        are_classes_list.append(0)

    # close webdriver
    driver.quit()

# create dataframe to narrow down majors to ones that have classes
classes_df = pd.DataFrame(
    {'Abbreviation': abb_list,
     'are_classes': are_classes_list
    })

# merge original df with classes df and drop majors w/o classes since we won't need to scrape those
df = pd.merge(df, classes_df, on='Abbreviation')
df = df[df.are_classes == 1]

# write df to csv
df.to_csv('majors_to_scrape.csv', index=False)