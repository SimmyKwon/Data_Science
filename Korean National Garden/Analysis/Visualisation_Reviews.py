# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:05:42 2023

@author: Minwoo Kwon
"""

#%%Import Packages
import pandas as pd
import itertools
from itertools import combinations as cb
from collections import Counter
from konlpy.tag import Okt, Kkma
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyvis.network import Network
import os, re

okt = Okt()
plt.rc('font', family = 'Malgun Gothic')

#%%Data Import

df_all = pd.read_csv('태화강_통합.csv')

df_blog = pd.read_csv('태화강_블로그_500.csv', index_col= 'Unnamed: 0')
df_insta = pd.read_csv('태화강_인스타_500.csv', index_col= 'Unnamed: 0')
df_cafe = pd.read_csv('태화강_네이버카페_500_수정.csv', index_col= 'Unnamed: 0')
df_twitter = pd.read_csv('태화강_트위터_500.csv', index_col= 'Unnamed: 0')

df_senti_KoB = pd.read_csv('태화강_KoBERT_긍부정.csv', index_col= 'Unnamed: 0') 
df_senti_KNU = pd.read_csv('태화강_KNU_긍부정.csv')

contents = df_senti_KoB['본문'].tolist()
stop_word = ['울산', '태화강', '국가', '정원', '바로', '정말', '진짜', '오늘', '정도', '보고', '여기', '생각', '시간', '광역시', '남구', '중구']

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

#%%Combine Sentiment

df_insta = pd.concat([df_insta, pd.Series(clean_all[:500]),df_senti_KoB.iloc[:500, :]], axis = 1).rename(columns = {0 : 'Token'})

df_twitter = pd.concat([df_twitter, pd.Series(clean_all[500:1000]),df_senti_KoB.iloc[500:1000, :]], axis = 1).rename(columns = {0 : 'Token'})
df_twitter.iloc[:500, 4:6] = df_twitter.iloc[500:1000, 4:6]; df_twitter= df_twitter.iloc[:500, :]

df_cafe = pd.concat([df_cafe, pd.Series(clean_all[1000:1500]),df_senti_KoB.iloc[1000:1500, :]], axis = 1).rename(columns = {0 : 'Token'})
df_cafe.iloc[:500, 4:6] = df_cafe.iloc[500:1000, 4:6]; df_cafe = df_cafe.iloc[:500, :]

df_blog = pd.concat([df_blog, pd.Series(clean_all[1500:]),df_senti_KoB.iloc[1500:, :]], axis = 1).rename(columns = {0 : 'Token'})
df_blog.iloc[:500, 5:7] = df_blog.iloc[500:1000, 5:7]; df_blog = df_blog.iloc[:500, :]

#%%Platform WordCloud

insta_token = list(itertools.chain.from_iterable(df_insta['Token'].tolist()))
twitter_token = list(itertools.chain.from_iterable(df_twitter['Token'].tolist()))
blog_token = list(itertools.chain.from_iterable(df_blog['Token'].tolist()))
cafe_token = list(itertools.chain.from_iterable(df_cafe['Token'].tolist()))

token_sns = [insta_token, twitter_token, blog_token, cafe_token]
sns_names = [name for name, var in globals().items() if isinstance(var, list) and var in token_sns]

for h, (tk, name) in enumerate(zip(token_sns, sns_names)):
    
    counts = Counter(tk)
    wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
                   background_color = "white", max_font_size=300, max_words=30)
    wc.generate_from_frequencies(counts)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    wc.to_file(f'wc_{h}_{name}.png')

#%%

df_all = pd.concat([df_all, pd.Series(clean_all),df_senti_KoB.iloc[:,1]], axis = 1).rename(columns = {0 : 'Token'})

all_list = list(itertools.chain.from_iterable(df_all['Token'].tolist()))

all_counts = Counter(all_list)
wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
               background_color = "white", max_font_size=300, max_words=50)
wc.generate_from_frequencies(all_counts)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
wc.to_file('전체 워드클라우드.png')

#%%Classification

insta_pos = df_insta[df_insta['감성'] == 1]
insta_neg = df_insta[df_insta['감성'] == 0]
print('인스타그램 긍정 비율은: ' + str(100 * len(insta_pos)/500) + '%')

twitter_pos = df_twitter[df_twitter['감성'] == 1]
twitter_neg = df_twitter[df_twitter['감성'] == 0]
print('트위터 긍정 비율은: ' + str(100 * len(twitter_pos)/500) + '%')

cafe_pos = df_cafe[df_cafe['감성'] == 1]
cafe_neg = df_cafe[df_cafe['감성'] == 0]
print('카페 긍정 비율은: ' + str(100 * len(cafe_pos)/500) + '%')

blog_pos = df_blog[df_blog['감성'] == 1]
blog_neg = df_blog[df_blog['감성'] == 0]
print('블로그 긍정 비율은: ' + str(100 * len(blog_pos)/500) + '%')

#%%WC prep for senti & source

insta_pos_token = list(itertools.chain.from_iterable(insta_pos['Token'].tolist()))
insta_neg_token = list(itertools.chain.from_iterable(insta_neg['Token'].tolist()))

twitter_pos_token = list(itertools.chain.from_iterable(twitter_pos['Token'].tolist()))
twitter_neg_token = list(itertools.chain.from_iterable(twitter_neg['Token'].tolist()))

cafe_pos_token = list(itertools.chain.from_iterable(cafe_pos['Token'].tolist()))
cafe_neg_token = list(itertools.chain.from_iterable(cafe_neg['Token'].tolist()))

blog_pos_token = list(itertools.chain.from_iterable(blog_pos['Token'].tolist()))
blog_neg_token = list(itertools.chain.from_iterable(blog_neg['Token'].tolist()))

all_tokens = [insta_pos_token, insta_neg_token, twitter_pos_token, twitter_neg_token, cafe_pos_token, cafe_neg_token, blog_pos_token, blog_neg_token]
list_names = [name for name, var in globals().items() if isinstance(var, list) and var in all_tokens]

#%%WordCloud creation for sentiment & source
for k, (tok, nm) in enumerate(zip(all_tokens, list_names)):
    
    counts = Counter(tok)
    wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
                   background_color = "white", max_font_size=300, max_words=20)
    wc.generate_from_frequencies(counts)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    wc.to_file(f'wc_{k}_{nm}.png')

#%%Whole positive and negative wordcloud



all_pos = df_all[df_all['감성'] == 1]
all_neg = df_all[df_all['감성'] == 0]

#%%positive & negative

pos_list = list(itertools.chain.from_iterable(all_pos['Token']))
neg_list = list(itertools.chain.from_iterable(all_neg['Token']))

pos_counts = Counter(pos_list)
wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
               background_color = "white", max_font_size=300, max_words=20)
wc.generate_from_frequencies(pos_counts)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
wc.to_file('긍정 전체.png')

neg_counts = Counter(neg_list)
wc = WordCloud(font_path ='font/malgun.ttf', width = 800, height = 600, 
               background_color = "white", max_font_size=300, max_words=20)
wc.generate_from_frequencies(neg_counts)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
wc.to_file('부정 전체.png')

