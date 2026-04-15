#%% ### Import Packages

import pandas as pd
import numpy as np
import itertools
from itertools import combinations as cb
from collections import Counter
from konlpy.tag import Okt, Kkma
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyvis.network import Network
import os, re
import datetime

okt = Okt()
plt.rc('font', family = 'Malgun Gothic')

#%% ### Import Dataset

# 데이터 불러오기
instar_raw = pd.read_csv('태화강_인스타_500.csv')
instar_raw['platform'] = '인스타그램'
print('인스타그램 : ', instar_raw.shape)

twitter_raw = pd.read_csv('태화강_트위터_500.csv')
twitter_raw['platform'] = '트위터'
print('트위터 : ', twitter_raw.shape)

naver_cafe_raw = pd.read_csv('태화강_네이버카페_500_수정.csv')
naver_cafe_raw['platform'] = '네이버카페'
print('네이버카페 : ', naver_cafe_raw.shape)

naver_blog_raw = pd.read_csv('태화강_블로그_500.csv')
naver_blog_raw['platform'] = '블로그'

for i, date in enumerate(naver_blog_raw['dates']):
    dt_obj = datetime.datetime.strptime(date.replace(' ',''), '%Y.%m.%d')  # 문자열을 datetime 객체로 변환
    formatted_date = dt_obj.strftime('%Y%m%d')  # 문자열로 변환
    naver_blog_raw.loc[i, 'dates'] = formatted_date
print('네이버블로그 : ', naver_blog_raw.shape)

# 데이터 통합
df = pd.concat([instar_raw, twitter_raw, naver_cafe_raw, naver_blog_raw], axis = 0, ignore_index=True)

# 제목 + 본문, 결측치 제거
df['titles'].fillna('', inplace = True)
df['contents_all'] = df['titles'].astype(str) + ' ' + df['contents'].astype(str)
df = df[df['contents_all'].str.strip() != '']
df = df.drop(columns=['Unnamed: 0','contents','titles','tags'])
print('본문 결측행 제거 후, 행 개수는 ', df.shape[0])
print(df.head(3))

# 통합 df 저장
os.chdir('..')
df.to_csv('태화강_통합.csv', index = False, encoding='utf-8-sig')

contents = df['contents_all']


#%% Stopwords, Noun

# 추가 불용어
stop_word = ['울산', '태화강', '국가', '정원', '바로', '정말', '진짜', '오늘', '정도', '보고', '여기', '생각', '시간', '광역시', '남구', '중구']

# 전체 전처리 결과 담을 공간
clean_all = []

for i, content in enumerate(contents):

    # 각 내용별 전처리 결과 담을 공간
    clean_word_in_one_content = []

    # 토큰화, 어간추출 , 정규화
    words = okt.pos(content, stem = True, norm = True)

    for word in words:
        if word[0] not in stop_word:    # 불용어 제거
            if word[1] in ['Noun']:    # 명사 추출
                    if len(word[0]) > 1 :   # 1글자 제거
                        clean_word_in_one_content.append(word[0])
    clean_all.append(clean_word_in_one_content)

    # print(f'{i} 번째 본문 : {contents[i]}')
    # print(f'{i} 번째 본문 처리후 : {clean_all[i]}')


# %% ### WordCloud

# 파일 내 단어 모두 통합
words_list_all = list(itertools.chain.from_iterable(clean_all))
count = Counter(words_list_all)

# 정렬하고 csv로 저장
wc_count = pd.DataFrame.from_dict(count, orient='index').reset_index()
wc_count.rename(columns={'index': 'word', 0: 'freq'}, inplace=True)
wc_count.sort_values(by='freq', ascending=False, inplace=True)
wc_count.to_csv('WordCloud.csv', encoding='utf-8-sig', index=False)

# 상위 키워드 확인
wc_n = 20
print(f'자주 출현하는 단어 {wc_n}개는 다음과 같습니다. \n {count.most_common(wc_n)}')

# 빈도 막대 그래프
wc_n = 10
wc_top = wc_count[:10].sort_values(by= 'freq',ascending= False)
fig, ax = plt.subplots(figsize=(12,8))
ax.barh(wc_top['word'], wc_top['freq'], color='green')
ax.set_xlabel('단어')
ax.set_ylabel('빈도수')
plt.xticks(rotation=45, ha='right')
plt.axis('off')
plt.savefig(f'빈도막대그래프{wc_n}.png')
# plt.show()

wc_n = 20
wc_top = wc_count[:20]
fig, ax = plt.subplots(figsize=(12,8))
ax.bar(wc_top['word'], wc_top['freq'], color='green')
ax.set_xlabel('단어')
ax.set_ylabel('빈도수')
plt.xticks(rotation=45, ha='right')
plt.savefig(f'빈도막대그래프{wc_n}.png')

# 시각화
wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
               background_color = "white", max_font_size=300, max_words=50)
wc.generate_from_frequencies(count)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
wc.to_file('wc.png')
# plt.show()
    

#%% ### Network

# 연관어 조합 딕셔너리 생성
network_count = {}

# 단어조합 카운팅
for words in clean_all:
    words = list(set(words))    # 줄별 중복 제거
    for idx, a in enumerate(words):
        for b in words[idx + 1:]:    # 한 단어씩 비교
            if a == b:
                continue
            elif a > b:
                network_count[b, a] = network_count.get((b, a), 0) + 1
            else:
                network_count[a, b] = network_count.get((a, b), 0) + 1

# 데이터프레임 형태로 변환
network = pd.DataFrame.from_dict(network_count, orient='index').reset_index()
network.rename(columns={'index' : 'words',0: 'freq'}, inplace=True)
network['freq'].astype('int')

# 정렬하고 csv로 저장
network.sort_values(by='freq', ascending=False, inplace = True)
network['node1'] = [idx[0] for idx in network['words']]
network['node2'] = [idx[1] for idx in network['words']]
network.to_csv('Network.csv', encoding='utf-8-sig', index=False)

# 상위 키워드 조합 확인
nx_n = 20
nx_top = network[:nx_n]   # 일부만 추출하여 시각화
print(f'자주 출현하는 단어 {20}개는 다음과 같습니다. \n {nx_top}')

# 시각화
# Network 객체 생성
net = Network(width='800px', height='800px', bgcolor='white', font_color='black')

# edge, node 생성
for node1, node2, freq in zip(nx_top['node1'], nx_top['node2'], nx_top['freq']):
    net.add_node(node1)
    net.add_node(node2)
    net.add_edge(node1, node2, value=freq)

# 시각화
net.save_graph(f'network{nx_n}.html')

nx_n = 30
nx_top = network[:30]   
net = Network(width='800px', height='800px', bgcolor='white', font_color='black')
for node1, node2, freq in zip(nx_top['node1'], nx_top['node2'], nx_top['freq']):
    net.add_node(node1)
    net.add_node(node2)
    net.add_edge(node1, node2, value=freq)
net.save_graph(f'network{nx_n}.html')