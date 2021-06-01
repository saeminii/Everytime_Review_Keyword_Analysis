import pandas as pd
import numpy as np
import difflib
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
import re

from wordcloud import WordCloud
from collections import Counter
from os import path

# 한글 폰트를 가져와야 함.(본인 컴퓨터에서 가져오면 됨.)
FONT_PATH = 'C:/windows/Fonts/a드림고딕4.ttf'


# 한글폰트 지정
plt.rc("font", family = "Malgun Gothic")
plt.rc("axes", unicode_minus = False)


# 스타일 지정
plt.style.use(['seaborn'])


import konlpy
import gc

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from krwordrank.sentence import summarize_with_sentences
from krwordrank.word import KRWordRank
from krwordrank.hangle import normalize

import MeCab
import networkx as nx
import os
plt.interactive(False)

##########################################################################################
##########################################################################################

class Utils:
    """static class for utils used in the analysis"""

    @staticmethod
    def get_lecture_names():
        lectures = ['영화의이해', '음악사', '디자인과문화', '현대사회와심리학', '이상행동의심리', '철학과윤리', '행복의 과학',
 '문명과질병', '재산거래와법','헌법의이해', '범죄와형벌', '국제사회와법이야기', '현대사회와경제', '세계평화와발전의국제관계',
 '수학과금융사회', '빅데이터와지식탐사', '우주의이해', '기후와문명', '건강과운동', '활과리라', '인간의삶과역사속의미생물',
 '융합시대의디지털멀티미디어','디지털방송의진화', '인공지능의이해와활용']
          
        return lectures


    @staticmethod
    def name_check(name):
        """강의명이 정확한지 확인 및 반환"""

        lecture_names = Utils.get_lecture_names()
        if name in lecture_names:
            return name
        else:
            # 단순 오타일 경우 강의명 재확인
            suggestion = []
            for true_name in lecture_names:
                similarity = difflib.SequenceMatcher(None, name, true_name).ratio()
                if similarity > 0.5:
                    suggestion.append(true_name)

            print("해당 강의가 없습니다.")
            print("혹시 이 강의인가요?: {}".format(suggestion))
            print()
            return "unmatch"


    @staticmethod
    def get_stopwords():
        stopwords =  ['강의', '교수님', '교수', '수업','내용', '강의', '시험', '과제', '학점', '해주시', '때문', '하루', '이번', '만큼','자체',
            '학생','학기','연대','평소','이번','제가','본인','사람','시간', '문제', '윤소연', '정도', '현정', '얘기', '느낌', '니다', '이다', '그렇',
             '경우', '나름', '고리', '부분', '때문', '한데' ,'나다', '라고', '진짜', '동안', '대로', '와서', '하다', '되다', '같다', '그러']

        return stopwords


    @staticmethod
    def Loading():
        """로딩"""
        print("조회중입니다", end='')

        for i in range(3):
            time.sleep(0.5)
            print(".", end='')
        print()
        print()


    @staticmethod
    def Borderline():
        """줄바꿈"""
        print()
        print(
            "##########################################################################################\n##########################################################################################")


##########################################################################################
##########################################################################################

class Analysis_Review(Utils):
    """강의별 키워드 분석"""

    def __init__(self, name):
        self.name = name
        self.data = pd.read_csv('everytime_review_pre.csv')
        self.stop_words = self.get_stopwords()
        
        
    def review_cloud(self):
        df = self.data[self.data['lecture'] == self.name]
        contents = list(df['review'])
        contents = ' '.join(contents)


        # 형태소 분석기
        komoran = konlpy.tag.Komoran(userdic='사용자사전.txt')
        komoran_nouns = komoran.nouns(contents)
        words_list = [w for w in komoran_nouns if len(w) != 1]

        # 실질 형태소 필터링
        dic_c = Counter(words_list)
        for k in list(dic_c.keys()):
            if dic_c[k] < 8:
                del dic_c[k]
            elif k in self.stop_words:
                del dic_c[k]
            else:
                continue
        #word_list = list(dic_c.keys())


        wordcloud = WordCloud(max_font_size=75, relative_scaling=.8, font_path=FONT_PATH, 
                              background_color='white', max_words = 200)
        
        # 전체 text만 넣어서 단어구름 만들기 
        #contents= ' '.join(list(dic_c.keys()))
        #wordcloud = wordcloud.generate_from_text(contents)
        
        # 빈도수 정보를 함께 넣어 빈도수에 따라 크기 달라지는 구름 만들기
        wordcloud = wordcloud.generate_from_frequencies(dic_c)
        #wordcloud.to_file(f'./image/wordcloud_{lecture}.jpg')

        plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        
    def review_cloud_pos(self):
        df = self.data[(self.data['lecture'] == self.name)&(self.data['stars'].isin([4,5]))]
        contents = list(df['review'])
        contents = ' '.join(contents)


        # 형태소 분석기
        komoran = konlpy.tag.Komoran(userdic='사용자사전.txt')
        komoran_nouns = komoran.nouns(contents)
        words_list = [w for w in komoran_nouns if len(w) != 1]

        # 실질 형태소 필터링 및 불용어 제거
        dic_c = Counter(words_list)
        for k in list(dic_c.keys()):
            if dic_c[k] < 6:
                del dic_c[k]
            elif k in self.stop_words:
                del dic_c[k]
            else:
                continue
        #word_list = list(dic_c.keys())


        wordcloud = WordCloud(max_font_size=75, relative_scaling=.8, font_path=FONT_PATH, 
                              background_color='white', max_words = 180, colormap = "Blues")
        wordcloud = wordcloud.generate_from_frequencies(dic_c)

        plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    
    
    # 평점 1,2 점 강의만 워드클라우드
    def review_cloud_neg(self):
        df = self.data[(self.data['lecture'] == self.name)&(self.data['stars'].isin([1,2]))]
        contents = list(df['review'])
        contents = ' '.join(contents)


        # 형태소 분석기
        komoran = konlpy.tag.Komoran(userdic='사용자사전.txt')
        komoran_nouns = komoran.nouns(contents)
        words_list = [w for w in komoran_nouns if len(w) != 1]

        # 실질 형태소 필터링 및 불용어 제거
        dic_c = Counter(words_list)
        for k in list(dic_c.keys()):
            if dic_c[k] < 2:
                del dic_c[k]
            elif k in self.stop_words:
                del dic_c[k]
            else:
                continue
        #word_list = list(dic_c.keys())


        wordcloud = WordCloud(max_font_size=75, relative_scaling=.8, font_path=FONT_PATH, 
                              background_color='white',  max_words = 180, colormap = 'flare')
        wordcloud = wordcloud.generate_from_frequencies(dic_c)

        plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


        
    # 원형복원
    def get_lemma(self, text):
        stop_words = self.stop_words
        tokenizer = MeCab.Tagger()
        parsed = tokenizer.parse(text)
        # print(parsed)
        word_tag = [w for w in parsed.split("\n")]
        pos = []
        tags = ["NNG", "NNP", "VV", "VA", "VCP", 'VCN', 'XR']

        for word_ in word_tag[:-2]:
            word = word_.split('\t')  # ['아버지', 'NNG,*,F,아버지,*,*,*,*']
            tag = word[1].split(",")  # ['EC', '*', 'F', '는다', '*', '*', '*', '*']
            if ('+' in tag[0]):  # 단어가 여러 형태소로 구성된 경우
                if ('VV' in tag[0] or 'VA' in tag[0]):
                    t = tag[-1].split('/')[0]
                    if t not in stop_words:
                        pos.append(t+"다")
                elif 'VX' in tag[0]:
                    t = tag[-1].split('/')[0]
                    if t not in stop_words:
                        pos.append(t)
            elif ('VV' in tag[0] or 'VA' in tag[0]):
                if (word[0] + "다") not in stop_words:
                    pos.append(word[0]+"다")
            elif ((tag[0] in tags) and (word[0] not in stop_words)):
                pos.append(word[0])
        return pos


       
    def keyword_network(self, keyword):

        # 강의명, 키워드
        review_list = list(self.data[self.data['lecture'] == self.name]['review'])
        review_list = [r for r in review_list if keyword in r]
        #print(review_list)

        # 토크나이저
        tf = CountVectorizer(tokenizer=self.get_lemma, preprocessor=None, lowercase=False, max_features = 130)

        tdm = tf.fit_transform(review_list)
        words = tf.get_feature_names()

        # array
        tdm_arr = tdm.toarray()
        words_list = []

        for r in range(len(tdm_arr)):
            for x in range(len(tdm_arr[0])):
            # print(words[x])

                if (tdm_arr[r][x] > 0) and (words[x] != keyword):
                    words_list.append(words[x])


        #관계 지정하기
        relations = []

        #키워드 여러개를 위해서는 수정 필요
        for x in words_list:
            relations.append((keyword,x))


        c = Counter(relations)
        #c = Counter(relations).most_common(n=100) -> 상위 n개만
        contains = []
        degree = {}
        for k, v in c.items():
            if len(k[1]) == 1:
                continue
            contains.append(k)
            degree[k[1]] = v

        # degree를 조정하면서 포함되지 않는 키워드로 인한 오류 방지 : 포함 키워드 리스트 새로 만들기
        degree_key = list(set([k[0] for k in contains]))
        max_degree = max(c.values())
        print(degree_key)
        for k in degree_key:
            # 최대 빈도보다 높은 degree 부여하기
            degree[k] = max_degree + 20

        # degree scale
        Scaler = MinMaxScaler(feature_range=(1,40))
        scaled_degree = Scaler.fit_transform(np.array(list(degree.values())).reshape(-1,1))
        scaled_degree = [v[0] for v in scaled_degree.tolist()]

        plt.figure(figsize=(35,20))
        plt.axis('off')
        G1 = nx.Graph()
        G1.add_edges_from(contains, color='blue')


        #pos = nx.random_layout(G1)
        pos = nx.spring_layout(G1)
        #pos = nx.draw_circular(G1)

        nx.draw_networkx(G1
                     ,nodelist= degree.keys()
                     ,node_size=[v*1000 for v in scaled_degree]
                     ,alpha=0.7
                    ,font_family='NanumGothic'
                    ,font_size=25
                    ,edge_color='.5'
                    ,font_color='black'
                    ,node_color=list(scaled_degree)
                    ,cmap=plt.cm.YlGn
                     )
        # nx.draw_networkx_labels(G1, pos, font_family=font_name, font_size=20)
        plt.show()


        # 키워드 포함 핵심 문장 추출

        review_list = list(self.data[self.data['lecture'] == self.name]['review_before']) # 문장 구분을 위해 before 사용 (문장 기호 포함)
        review_list = [r for r in review_list if keyword in r]

        beta = 0.85    # PageRank의 decaying factor beta
        max_iter = 10
        penalty = lambda x:0 if (len(x) <= 200) else 1


        keywords, sents = summarize_with_sentences(
            review_list,
            penalty = penalty,
            stopwords = self.stop_words,
            diversity=0.5,
            num_keywords=130,
            num_keysents=5,
            verbose=False
        )

        for sent in sents:
            print(sent)



    def lecture_stars_rate(self):
        data = self.data[self.data['lecture'] == self.name]
        stars = data.groupby(['stars']).count()
        stars['sum'] = stars['category'].sum()
        stars['rate'] = stars['review']/stars['sum']

        sns.set_palette("YlOrBr", 10)
        plt.figure(figsize = (8,6))
        g = sns.barplot(
            data = stars,
            x = stars.index,
            y = stars.rate)
        for p in g.patches:
            left, bottom, width, height = p.get_bbox().bounds
            plt.annotate("%.f"%(height*100) + '%' , (left+width/2, height*1.01), ha = 'center',fontsize = 12)
        #plt.title(f'{lecture} : 평점 별 비율')
        plt.show()
        

               


    def script(self):

        Utils.Loading()
        Utils.Borderline()
        print("선택하신 강의는 [{}] 입니다.\n".format(self.name))

        choice = 0
        while choice != 4:
            print("어떤 것을 확인하시겠습니까?\n")
            print("1. 강의평 단어 구름")
            print("2. 키워드 검색")
            print("3. 평점 별 비율")
            print("4. 뒤로가기")
            try:
                choice = int(input(":"))
            except:
                choice = int(input(":"))

            if choice == 1:
                Utils.Borderline()
                cloud_choice = 0
                while cloud_choice != 4:
                    print("확인할 단어 구름을 선택해주세요\n")
                    print("1. 전체 강의평")
                    print("2. 평점 1,2 점 강의평")
                    print("3. 평점 4,5 점 강의평")
                    print("4. 뒤로가기")
                    try:
                        cloud_choice = int(input(":"))
                    except:
                        cloud_choice = int(input(":"))
                        
                    if cloud_choice == 1:
                        self.review_cloud()
                    elif cloud_choice == 2:
                        self.review_cloud_neg()
                    elif cloud_choice == 3:
                        self.review_cloud_pos()
                    elif cloud_choice == 4:
                        pass
                    else:
                        print('잘못된 입력입니다.')

                print()
                input("아무키나 누르세요.")
                Utils.Borderline()

            elif choice == 2:
                Utils.Borderline()
                keyword = input("궁금한 키워드를 입력하세요\n")
                keyword_map = self.keyword_network(keyword)
                print()
                input("아무키나 누르세요.")
                Utils.Borderline()

            elif choice == 3:
                Utils.Borderline()
                Utils.Loading()
                star_rate = self.lecture_stars_rate()
            
                print()
                input("아무키나 누르세요.")
                Utils.Borderline()


            elif choice == 4:
                pass

            else:
                print("잘못된 입력입니다.")


