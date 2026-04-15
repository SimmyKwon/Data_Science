# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:02:01 2023

@author: BAKorea
"""

#%%Import Packages

import pandas as pd
import numpy as np
import selenium.webdriver as wbd
from selenium.common.exceptions import NoSuchElementException as NSE
from selenium.common.exceptions import StaleElementReferenceException as SERE
from selenium.webdriver.common.keys import Keys
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

#driver = wbd.ChromiumEdge()
driver = wbd.Chrome('chromedriver.exe')
url = 'https://www.naver.com/'

driver.get(url)
driver.implicitly_wait(7)

Action = ActionChains(driver)

#%%Search Keywords
srch_2 = '태화강국가정원'

#%%Go to search bar and  click it

search_bar = driver.find_element(By.XPATH, '//*[@id="query"]')
search_bar.click()

method_2 = '네이버 블로그'
search_bar.send_keys(method_2)
search_bar.send_keys(Keys.ENTER)

Blog_link = driver.find_element(By.XPATH, '//*[@id="main_pack"]/section[1]/div/div/div[1]/div/div[2]/a')
Blog_link.click()

window_handles = driver.window_handles
driver.close()
driver.switch_to.window(window_handles[1])

Blog_search_bar = driver.find_element(By.XPATH, '//*[@id="header"]/div[1]/div/div[2]/form/fieldset/div/input')
Blog_search_bar.click()

Blog_search_bar.send_keys(srch_2)
Blog_search_bar.send_keys(Keys.ENTER)

time.sleep(1)

Recents = driver.find_element(By.XPATH, '//*[@id="content"]/section/div[1]/div[2]/div/span/a[2]/span')
Recents.click()

#%%Set the date of search time

Srch_Opt = driver.find_element(By.CLASS_NAME, 'search_option')
Srch_arrow = Srch_Opt.find_element(By.CLASS_NAME, 'icon_arrow')

Srch_arrow.click()

time.sleep(1)

Start_Date = '2023.04.10'
End_Date = '2023.04.21'
time.sleep(1)

Start_box = Srch_Opt.find_element(By.XPATH, '//*[@id="search_start_date"]')
End_box = Srch_Opt.find_element(By.XPATH, '//*[@id="search_end_date"]')

Start_box.click()
Start_box.send_keys(Start_Date)
Start_box.send_keys(Keys.ENTER)

End_box.click()
End_box.send_keys(End_Date)
End_box.send_keys(Keys.ENTER)

Srch_But = Srch_Opt.find_element(By.XPATH, '//*[@id="periodSearch"]')
Srch_But.click()

#%%Prelims Pt.1

Blog_Dat = pd.DataFrame()
Titles = []
Contents = []
Hashtags = []
Dates = []

#%%Prelims Pt.2

Post_area = driver.find_element(By.CLASS_NAME, 'area_list_search')

Post_items = Post_area.find_elements(By.CLASS_NAME, 'list_search_post')

Post_per_page = len(Post_items)

tot_pages = int(500/Post_per_page) + 2


#%%Crawler

def Crawl_Blog(link):
    
    txt_temp = []  
    hsh_temp = []
    time.sleep(1.5)
    
    driver.get(link)
    
    time.sleep(1.5)
    
    driver.switch_to.frame('mainFrame')
    
    try:
        date = driver.find_element(By.CLASS_NAME, 'se-component-content').find_element(By.CLASS_NAME, 'se_publishDate').text
        body = driver.find_element(By.CLASS_NAME, 'se-main-container')
        mains = body.find_elements(By.CLASS_NAME, 'se-text-paragraph')
        
    except NSE:
        date = driver.find_element(By.CLASS_NAME, 'blog2_container').find_element(By.CLASS_NAME, 'se_publishDate').text
        body = driver.find_elements(By.CLASS_NAME, 'se_component_wrap')[1]
        mains = body.find_elements(By.CLASS_NAME, 'se_paragraph')
    
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
                Contents.append(f'No Contents for {title}_{p}')
                
        except:
            Contents.append(f'No Contents for {title}_{p}')
            
    try:   
        hashs = driver.find_element(By.CLASS_NAME, 'wrap_tag').find_elements(By.TAG_NAME, 'a')
    except:
        hashs = '-'
    
    
    for k in hashs:
        if len(k.text)> 0:
            hsh_temp.append(k.text)
            
    hsh_tot = " ".join(hsh_temp)
    Hashtags.append(hsh_tot)
        
    Dates.append(date)

    txt_temp.clear()
    hsh_temp.clear()
    
    driver.back()

#%%Boss: Crawl

p = 1

while True:
    
    time.sleep(2)
    print(f'네이버 블로그 {srch_2} 관련 게시물 {p} 페이지 크롤링을 시작합니다.')
    
    Post_area = driver.find_element(By.CLASS_NAME, 'area_list_search')
    Post_items = Post_area.find_elements(By.CLASS_NAME, 'list_search_post')
    
    Link_Tot = []
    
    for pp in Post_items:
        
        Url = pp.find_element(By.CLASS_NAME, 'desc').find_element(By.TAG_NAME, 'a').get_attribute('href')
        Link_Tot.append(Url)
    
    for ix, pg in enumerate(Post_items):
        
        try:
            
            link = pg.find_element(By.CLASS_NAME, 'desc').find_element(By.TAG_NAME, 'a').get_attribute('href')
            title = pg.find_element(By.CLASS_NAME, 'desc').find_element(By.CLASS_NAME, 'title').text
            Titles.append(title)
            
            Crawl_Blog(Link_Tot[ix])
            time.sleep(1)
            
        except SERE:
            pg = driver.find_elements(By.CLASS_NAME, 'list_search_post')[ix]
            link = pg.find_element(By.CLASS_NAME, 'desc').find_element(By.TAG_NAME, 'a').get_attribute('href')
            title = pg.find_element(By.CLASS_NAME, 'desc').find_element(By.CLASS_NAME, 'title').text
            Titles.append(title)
            
            Crawl_Blog(Link_Tot[ix])
            time.sleep(1)
            
        
        Post_area = driver.find_element(By.CLASS_NAME, 'area_list_search')
        Post_items = Post_area.find_elements(By.CLASS_NAME, 'list_search_post')
        
        if ix < Post_per_page - 1:
            driver.execute_script("arguments[0].scrollIntoView();", Post_items[ix+1])
            
        elif ix == Post_per_page - 1:
            
            if p < tot_pages:
                if p % 10 != 0:
                    next_button = driver.find_element(By.CLASS_NAME, 'pagination').find_element(By.LINK_TEXT, f'{p+1}')
                    Action.move_to_element(next_button).click().perform()

                    
                elif p % 10 == 0:
                    next_sec = driver.find_element(By.CLASS_NAME, 'pagination').find_element(By.CLASS_NAME, 'button_next')
                    Action.move_to_element(next_sec).click().perform()
                
            elif p == tot_pages:
                print('네이버 블로그 크롤링이 종료되었습니다.')
                driver.close()
                break
    
    p += 1
    
#%%Turn into Dataframe

Blog_Dat['titles'] = Titles
Blog_Dat['contents'] = Contents
Blog_Dat['dates'] = Dates
Blog_Dat['tags'] = Hashtags

Blog_Dat_fin = Blog_Dat.drop_duplicates(subset='contents').reset_index(drop = True)

Blog_Dat_Sav = Blog_Dat_fin.iloc[:500,:]

Blog_Dat_Sav.to_csv('네이버 블로그_500개.csv', encoding = 'utf-8')
