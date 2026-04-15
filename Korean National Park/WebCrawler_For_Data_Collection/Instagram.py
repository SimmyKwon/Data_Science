
# %%
# 인스타그램 크롤링

# XPATH가 변해서 CLASS_NAME 활용하여 수집
# 자주 로그인 , 로그아웃 하면 로그인 제한 걸림 
# -> 대화형 인터프리터로 테스트하며 코딩

# 환경 구성
import pandas as pd
import time, re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs
import requests as rq
from datetime import datetime

# 크롬 브라우저 오픈
browser = webdriver.Chrome('chromedriver.exe')
url = 'https://www.instagram.com/accounts/login'
browser.get(url)

# 페이지 열릴 때까지 대기 시간 부여(단위 초)
browser.implicitly_wait(10)

### 개인정보니까 가리고 줄것 ###
# 아이디, 비밀번호 설정
---id = "jaehyeok_nam"
---pw = "Angrybird1"

# 아이디, 비밀번호 입력
input = browser.find_elements(By.TAG_NAME,'input')
input[0].send_keys(id)
input[1].send_keys(pw)

# 엔터
input[1].send_keys(Keys.RETURN)
time.sleep(2)

# 팝업 - 나중에 하기 클릭
browser.find_element(By.CLASS_NAME, '_ac8f').click()  # 로그인 정보 저장 여부
time.sleep(3)
browser.find_element(By.CLASS_NAME, '_a9--._a9_1').click()   # 알림 설정
time.sleep(3)

# 검색 단어 지정
search_word = '제주여행'

# 수집 목표 개수
n = 3000

# url 생성 및 열기
url_search = f'https://www.instagram.com/explore/tags/{search_word}/'
browser.get(url_search)
browser.implicitly_wait(10)

# 저장 변수 생성
contents = []
tags = []
dates = []

print('수집 준비 완료!')

# %%

# 게시물 리스트 가져오기
posts = browser.find_elements(By.CLASS_NAME,'_aagw')

# 9개의 인기 게시물 다음 최근 게시물 클릭
posts[10].click()
time.sleep(1.5)

for i in range(1,n+1):
    # 다음 게시물
    print(i, '번째 게시물 수집 중')
    next = browser.find_element(By.CLASS_NAME, '_aaqg._aaqh')
    next.click()
    time.sleep(1)

    # 본문
    try : 
        content_html = browser.find_element(By.CLASS_NAME, '_aacl._aaco._aacu._aacx._aad7._aade')
        content = content_html.text
        contents.append(content)
    except : # 내용이 없는 경우
        content = ''
        contents.append(content)

    # 해시태그, 본문 내에서 탐색
    tag = re.findall(r'#+[\S]+', content)
    tags.append(tag)

    # 작성일자
    date_raw = browser.find_element(By.CLASS_NAME, '_aaqe').get_attribute('title')  # 4월 14, 2023 형태로 수집됨
    date = datetime.strptime(date_raw, "%m월 %d, %Y")  # datetime 객체로 변환
    date_formatted = date.strftime("%Y%m%d")  # 20230414로 변환
    dates.append(date_formatted)

# 게시물 닫기
browser.find_element(By.CLASS_NAME, 'x78zum5.x6s0dn4.xl56j7k.xdt5ytf').click()

# 데이터 통합
df = pd.DataFrame([contents, tags, dates]).T
df.columns = ['contents', 'tags', 'dates']
df.drop_duplicates(subset = ['contents', 'dates'], inplace = True, ignore_index=True)  # 도배글 제거
print(df)
df.to_csv("인스타그램test.csv", encoding='utf-8-sig')

# %%
