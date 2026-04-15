import pandas as pd
import numpy as np
import itertools
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os, re, copy

#%%Import Dictionary
Dict = pd.read_csv('SentiWord_Dict.txt', delimiter = '\t', header= None).rename(columns = {0: 'Word', 1: 'Score'})
Dict.value_counts('Score')
Dict_list = Dict['Word'].tolist()

okt = Okt()

def KNU():
    # 토큰화 진행
    df['Tokens_Okt'] = ''
    for i, content in enumerate(contents):
        df['Tokens_Okt'][i] = okt.morphs(str(content), norm = True, stem = True)
    
    print('토큰화 완료')

    # 감성분석 함수
    # 시간 복잡도: O(n), 각 행별로 리스트에서 단어들을 포함하는지 확인하기
    def scoring(text):
        txt_score = 0
        for word in text:
            if Dict['Word'].isin([word]).any():
                sc = Dict.loc[Dict['Word'] == word, 'Score'].values[0]
                txt_score += sc
        
        if np.isnan(txt_score):
            txt_score = 0
        
        txt_score = int(txt_score)
        return txt_score

    # 감성분석 진행
    Tokens_Okt = df['Tokens_Okt'].tolist()
    df['Okt_Score'] = ''

    for k, toekn in enumerate(Tokens_Okt):
        df['Okt_Score'][k] = scoring(toekn)
    
    print('감성분석 완료')

    # KNU Okt_Score -> 긍정 부정 변환
    def pos_neg(num):
        if num > 0:
            return 1
        elif num < 0:
            return 0
        else:
            return '중립'
        
    df['KNU_label'] = df['Okt_Score'].apply(pos_neg)
    print('긍부정 변환 완료')

 
# 데이터셋에 적용
df = pd.read_csv('태화강_통합.csv', encoding='utf-8-sig')
contents = df['contents_all']
KNU()
df.to_csv('태화강_KNU_긍부정.csv', encoding= 'utf-8-sig', index = False)

df = pd.read_csv('ratings_test.txt', delimiter = '\t')
contents = df['document']
KNU()
df.to_csv('ratings_test_KNU_긍부정.csv', encoding= 'utf-8-sig', index = False)


