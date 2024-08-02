from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# url of the webpage you want to scrape
year = 23
major = 'COM+SCI'
course_url = 'https://catalog.registrar.ucla.edu/major/2023/ComputerScienceBS?siteYear=2023'

# function to initialize selenium webdriver
def init_driver():
    options = webdriver.ChromeOptions()
    # Add any options you need, for example, to run headless
    #options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

def make_request(url):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, 'lxml')

    return soup

# function to scrape the specified JS path
def scrape_path():
    driver = init_driver()
    driver.get(course_url)

    time.sleep(5)

    #clicks major requirements button
    major_req = WebDriverWait(driver, 10).until(
         EC.presence_of_element_located((By.CSS_SELECTOR, '#smenu > ul > li:nth-child(4) > a > div'))
    )
    major_req.click()

    time.sleep(5)

    #Shadow Root exploration
    #clicks expand button
    
    # expand_button = WebDriverWait(driver, 10).until(
    #      EC.presence_of_element_located((By.CSS_SELECTOR, '#b6a2e9bd-1373-4df3-9f87-92c73ed0c7b6_expandAll > button'))
    # )

    #shadow_root_script = 'return document.querySelector("#\38 476b1a4-e2f8-4529-8ccc-eda94974a56d_expandAll > button").shadowRoot'
    # shadow_root = driver.execute_script(shadow_root_script)
    # print("executed shadow root")
    # print(shadow_root)

    # Wait for the shadow host element and then the shadow root element 

    # class presence_of_shadow_element(): 
    #     def init(self, host_selector, shadow_selector): 
    #         self.host_selector = host_selector 
    #         self.shadow_selector = shadow_selector

    # shadow_host_selector = "ucla-sa-soc-app" 
    # shadow_element_selector = "#b6a2e9bd-1373-4df3-9f87-92c73ed0c7b6_expandAll > button" # Use the custom expected condition to wait for the shadow element 
    # shadow_element = WebDriverWait(driver, 10).until(presence_of_shadow_element(shadow_host_selector, shadow_element_selector) )
    # print(shadow_element)

    # find the element within the shadow root and click to expand class info
    # element_within_shadow_root = shadow_root.find_element(By.CSS_SELECTOR, '#expandAll')
    # element_within_shadow_root.click()

    #attempted to click down button; did not work
    expand_button = WebDriverWait(driver, 10).until(
          EC.presence_of_element_located((By.CLASS_NAME, 'class="material-icons md-24    css-oxi151-MaterialIcon--IconWrapper e1lj33n40"'))
     )
    expand_button.click()

    #using XPATH
    # expand_button = WebDriverWait(driver, 10).until(
    #      EC.presence_of_element_located((By.XPATH, '//*[@id="b6a2e9bd-1373-4df3-9f87-92c73ed0c7b6_expandAll"]/button'))
    # )

    # expand_button = WebDriverWait(driver, 10).until(
    #      EC.presence_of_element_located((By.CSS_SELECTOR, '#\38 476b1a4-e2f8-4529-8ccc-eda94974a56d_expandAll > button'))
    # )

    # expand_button = WebDriverWait(driver, 10).until(
    #      EC.presence_of_element_located((By.CLASS_NAME, 'css-1nb17rj-SLinkButton--CallToActionButton erbjvpb0'))
    # )

    expand_button.click()

    driver.quit()
        
# soup = make_request(course_url)
# with open('output.html', 'w', encoding='utf-8') as file:
#     file.write(soup.prettify())
scrape_path()

