# -*- coding: utf-8 -*-
"""
Spyder Editor

Made by Minwoo Kwon
"""
#%%Import Packages

import pandas as pd
import numpy as np
import selenium.webdriver as wbd
from selenium.common.exceptions import NoSuchElementException as NSE
from selenium.common.exceptions import StaleElementReferenceException as SERE
from selenium.common.exceptions import UnexpectedAlertPresentException as UAPE
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys
import bs4
from selenium.webdriver.support.ui import WebDriverWait as wdw
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import time

#%%Open browsers
#options = wbd.ChromeOptions()
#options.use_chromium = True
#options.add_argument("--disable-extensions")
#options.add_argument("--disable-popup-blocking")
#options.add_argument("--disable-default-apps")

driver = wbd.ChromiumEdge()
#driver = wbd.Chrome('chromedriver.exe')
url = 'https://www.naver.com/'

driver.get(url)
driver.implicitly_wait(7)

Action = ActionChains(driver)

da = Alert(driver)

#%%Search Keywords

srch_1 = '태화강국가정원'

#%%Go to search bar and  click it

search_bar = driver.find_element(By.XPATH, '//*[@id="query"]')
search_bar.click()

method_1 = '네이버 카페'
search_bar.send_keys(method_1)
search_bar.send_keys(Keys.ENTER)

Cafe_link = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[2]/div/div[3]/a')
Cafe_link.click()

window_handles = driver.window_handles
driver.close()
driver.switch_to.window(window_handles[1])

time.sleep(2)

Cafe_search_bar = driver.find_element(By.XPATH, '//*[@id="header"]/div[1]/div/div[2]/form/fieldset/div/div/div[2]/input')
Cafe_search_bar.click()

Cafe_search_bar.send_keys(srch_1)
Cafe_search_bar.send_keys(Keys.ENTER)

time.sleep(1)

All_Cafe_posts = driver.find_element(By.XPATH, '//*[@id="mainContainer"]/div/div/div[2]/a[3]')
All_Cafe_posts.click()

Recents = driver.find_element(By.CLASS_NAME, 'SortTab').find_element(By.XPATH, '//*[@id="mainContainer"]/div/div/div[3]/div/div[2]/div/a[2]')
Recents.click()

#%%Set the date of search time

Period_Filter = driver.find_element(By.CLASS_NAME, 'FilterPeriodList').find_element(By.CLASS_NAME, 'filter_list')

SelfKeyIn = Period_Filter.find_element(By.CLASS_NAME, 'FormInputRadio[label-text = "직접입력"]')
SelfKeyIn.click()

Start_Date = '2023.01.01'
End_Date = '2023.04.21'
time.sleep(1)

Date_Input_Form = Period_Filter.find_element(By.CLASS_NAME, 'direct_insert')
Start_Input = Date_Input_Form.find_element(By.XPATH, '//*[@id="mainContainer"]/div/div/div[3]/div/div[1]/div[2]/ul/li[9]/div[1]/div/input')
End_Input = Date_Input_Form.find_element(By.XPATH, '//*[@id="mainContainer"]/div/div/div[3]/div/div[1]/div[2]/ul/li[9]/div[2]/div/input')

Start_Input.click()
Start_Input.send_keys(Start_Date)

End_Input.click()
End_Input.send_keys(End_Date)

Cafe_Srch_button = Date_Input_Form.find_element(By.CLASS_NAME, 'search_button')
Cafe_Srch_button.click()


#%%Prelims Pt.1

Cafe_Dat = pd.DataFrame()
Titles = []
Contents = []
Hashtags = []
Dates = []

#%%Prelims Pt.2

Article_area = driver.find_element(By.CLASS_NAME, 'article_list_area')

arti_items = Article_area.find_elements(By.CLASS_NAME, 'detail_area')

arti_per_page = len(arti_items)

tot_pages = int(500/arti_per_page) + 1

#%%Crawler

def Crawl_Cafe():
    
    txt_temp = []
    
    link = pg.find_element(By.TAG_NAME, 'a').get_attribute('href')
    
    time.sleep(2)
    driver.get(link)
    
    try:
        driver.switch_to.frame('cafe_main')
        time.sleep(2)
        try:
            date = driver.find_element(By.XPATH, '//*[@id="app"]/div/div/div[2]/div[1]/div[2]/div[2]/div[2]').find_element(By.CLASS_NAME, 'date').text.split(" ")[0]
        except NSE:
            date = driver.find_element(By.XPATH,'//*[@id="app"]/div/div/div[2]/div[1]/div[1]/div[2]/div[2]/span').text
        
        body = driver.find_element(By.ID, 'app').find_element(By.CLASS_NAME, 'article_viewer')
        try:
            mains = body.find_element(By.CLASS_NAME, 'se-main-container').find_elements(By.CLASS_NAME, 'se-text-paragraph')
        except NSE:
            mains = body.find_elements(By.CLASS_NAME, 'ContentRenderer')
    
        for x in mains:
            if len(x.text) > 0:
                txt_temp.append(x.text)
            
        txt_tot = " ".join(txt_temp)
    
        if len(txt_tot) != 0:
            Contents.append(txt_tot)
        
        elif len(txt_tot) == 0:
            try:
                txt_temp.clear()
                new_mains = body.find_elements(By.TAG_NAME, 'p')
        
                for y in new_mains:
                    if len(y.text)> 0:
                        txt_temp.append(y.text)
                    
                txt_tot_2 = " ".join(txt_temp)
            
                if len(txt_tot_2) > 0:
                    Contents.append(txt_tot_2)
            
                elif len(txt_tot_2) == 0:
                    Contents.append('-')
                
            except:
                Contents.append('-')
            
        Dates.append(date)
            
    except UAPE:
        
        driver.switch_to.alert()
        da.accept()
        driver.back()
        Dates.append('-')
        Contents.append('-')
        
        Article_area = driver.find_element(By.CLASS_NAME, 'article_list_area')
        arti_items = Article_area.find_elements(By.CLASS_NAME, 'detail_area')

    txt_temp.clear()
    
    driver.back()

#%%Boss: Crawl

p = 1

while True:
    
    time.sleep(2)
    print(f'네이버 카페 {srch_1} 관련 게시물 {p} 페이지 크롤링을 시작합니다.')
    
    Article_area = driver.find_element(By.CLASS_NAME, 'article_list_area')
    arti_items = Article_area.find_elements(By.CLASS_NAME, 'detail_area')

    for ix, pg in enumerate(arti_items):
        
        Action.move_to_element(arti_items[ix]).click().perform()
        #arti_items[ix].click()
        time.sleep(1)
        
        try:
            link = pg.find_element(By.TAG_NAME, 'a').get_attribute('href')
            title = pg.find_element(By.TAG_NAME, 'a').text
            Action.move_to_element(pg).click().perform()
            #pg.click()
            
            Crawl_Cafe()
            Titles.append(title)
            time.sleep(1)
            
        except SERE:
            pg = driver.find_elements(By.CLASS_NAME, 'detail_area')[ix]
            link = pg.find_element(By.TAG_NAME, 'a').get_attribute('href')
            title = pg.find_element(By.TAG_NAME, 'a').text
            Action.move_to_element(pg).click().perform()
            #pg.click()
            
            Crawl_Cafe()
            Titles.append(title)
            time.sleep(1)
        
        Article_area = driver.find_element(By.CLASS_NAME, 'article_list_area')
        arti_items = Article_area.find_elements(By.CLASS_NAME, 'detail_area')
        
        if ix < arti_per_page - 1:
            driver.execute_script("arguments[0].scrollIntoView();", arti_items[ix+1])

        elif ix == arti_per_page - 1:
            
            if p < tot_pages:
                next_button = driver.find_element(By.CLASS_NAME, 'SectionPagination').find_element(By.LINK_TEXT, f'{p+1}')
                Action.move_to_element(next_button).click().perform()
                #next_button.click()
                
            elif p == tot_pages:
                print('네이버 카페 크롤링이 종료되었습니다.')
                driver.close()
                break
    
    p += 1

#%%Comb

Cafe_Dat['titles'] = Titles
Cafe_Dat['contents'] = Contents
Cafe_Dat['dates'] = Dates

Cafe_Dat = Cafe_Dat.iloc[:500, :]

Cafe_Dat.to_csv('태화강_네이버카페_500_수정.csv', encoding = 'utf-8')

