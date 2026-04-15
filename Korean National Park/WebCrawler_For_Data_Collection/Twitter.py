# 트위터 크롤링

# API 키 받으려면 계정과 핸드폰 인증이 필요한데,
# 2022.12.16 이후로 한국 통신사 인증이 차단된 상태

# 환경 구성
import pandas as pd
import time, re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
import requests as rq
from datetime import datetime, timedelta

# 검색 단어 지정
search_word = '제주여행'

# 스크롤을 몇 번 내릴지 지정
# 스크롤할 때마다 xpath번호가 바뀌며, 스크롤당 생성되는 트윗 개수 또한 달라 스크롤 횟수 활용
# 대략적으로 스크롤 1번당 6-7개 트윗
scrolldown = 200

# 크롬 브라우저 오픈
browser = webdriver.Chrome('chromedriver.exe')
url = f'https://twitter.com/search?q={search_word}&src=typed_query&f=live'  # 단어 검색, 최신순 정렬 url
browser.get(url)

# 페이지 열릴 때까지 대기
browser.implicitly_wait(10)

# 알림 켜기 팝업 나올때까지 대기(팝업이 매우 늦게 나옴..)
time.sleep(20)
# 알림 켜기 - 나중에 버튼 클릭
browser.find_element(By.XPATH, '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div[2]/div/div[2]/div[2]/div[2]/div').click()

# 저장 변수 생성
contents = []
tags = []
dates = []

# 광고는 본문 xpath 주소가 다르고, datetime 정보가 없어 자동으로 필터링됨
for i in range(1,scrolldown + 1):
    # 답글이 있는 경우 xpath 번호가 변경되며 오류 발생 -> 전체 가져온 후 tweetText만 추출해보자
    # tweet 박스의 모든 요소를 가져옴
    contents_html = browser.find_elements(By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/section/div/div/div/div/div/article/div/div/div[2]/div[2]/div/div')

    for content_html in contents_html:
        try :
            # tweetText에 해당하는 html만 선택해서 텍스트 추출
            if content_html.get_attribute('data-testid') == 'tweetText':
                # 본문
                content = content_html.text
                contents.append(content)
        except : 
            pass

    # 작성일자
    dates_html = browser.find_elements(By.XPATH, '/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/section/div/div/div/div/div/article/div/div/div[2]/div[2]/div[1]/div/div[1]/div/div/div[2]/div/div[3]/a/time')
    for date in dates_html:
        date_raw = date.get_attribute('datetime')   #2023-04-13T18:04:56.000Z 형태로 수집됨
        date_utc = datetime.strptime(date_raw, '%Y-%m-%dT%H:%M:%S.%fZ')   # datetime 객체로 변환
        date_kst = date_utc + timedelta(hours=9)    # 9시간 시차 더해서 한국시간(KST)로 변경
        date_formatted = date_kst.strftime("%Y%m%d")   # 20230414로 변환
        dates.append(date_formatted) 

    # 스크롤 내려서 새 트윗들 로드
    # 최하단 픽셀 min-height 평균 7000/2 값을 크게 줬더니 일부가 안 긁히고 스킵됨
    browser.execute_script(f"window.scrollTo(0, 3500*{i});")  
    time.sleep(1.5)

browser.close()

'''    
해당 픽셀을 넘기면 트윗이 사라짐
0, 517, 1254, 1986, 2498, 
3275, 3414, 3601, 3998, 4117, 
4324, 4861, 5407, 5997, 6414, 
6692, 6839, 7436, 7558, 8181
트윗 길이 평균 397.05 => 400

min-height 조금씩은 변하지만 얼추 비슷한 수치
7761, 14654, 22353, 29891
스크롤을 조금씩 내리면서 수집 후 중복을 제거하는 방식 사용 
'''

# 본문 내에서 해시태그 추출
for content in contents:
    tag = re.findall(r'#+[\S]+', content)
    tags.append(tag)   

df = pd.DataFrame([contents, tags, dates]).T
df.columns = ['contents', 'tags', 'dates']

# 중복 크롤링 제거 + 도배글 제거
df.drop_duplicates(subset = ['contents', 'dates'], inplace = True, ignore_index=True)
print(df)
df.to_csv("트위터test.csv", encoding='utf-8-sig')

