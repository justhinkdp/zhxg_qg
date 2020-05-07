# encoding:utf-8

import numpy as np
import jieba

# vsm.py负责构建文本向量，用来训练模型/进行预测
def vsmbuild(clabel):
    # 处理第一层3个场景+others 场景的训练数据，转化为向量形式
    # word2id中key为关键词，value为关键词在word2id中的编号/位置（从0开始）
    # word2id_cat中key为关键词，value为关键词对应的场景
    if clabel == 1:
        word2id = {}
        word2id_cat = {}
        path = './data/'

        # 处理TF-IDF查找到的关键词，将其放在word2id这个dict中，key为关键词，value为1，2，3···
        for line in open(path + "keywords_single_250.txt", encoding='UTF-8'):
            for w in line.split():
                word2id[w.strip()] = len(word2id) # 相当于给每个feature.py提取的关键词编号1,2,3,4...，‘关键词1’：‘1’
        ct=0
        counts = [0, 0, 0, 0, 0]

        # 处理TF-IDF查找到的关键词，将其放在word2id_cat这个dict中，key为关键词，value为0/1/2/3/4，代表5个场景
        for line in open(path + "keywords_single_250.txt", encoding='UTF-8'):
            for w in line.split():
                word2id_cat[w.strip()] = ct
                counts[ct] += 1
            ct += 1

        # 处理训练数据，转化为向量形式
        data = []
        paths = [path + "level2new.txt", path + "level1new.txt", path + "level0new.txt", path + "level-1new.txt", path + "level-2new.txt"]

        for i in range(5):
            # print i,
            for line in open(paths[i], 'rb'):
                # tp为该条文本转化为的词向量，词向量长度为关键词长度+6，分别代表5个场景命中了多少个关键词+本条语句属于某一场景  e.g.  tp=[1,0,1,...,1,0,14,15,16,2]，最后四个之前表示命中了哪几个关键词，14表示命中14个场景0的关键词，15表示命中15个场景1的关键词，16表示命中16个场景2的关键词，2表示该文本属于场景2
                tp = [0] * (len(word2id) + 6)
    
                for w in jieba.cut(line):
                    if line == '\n':
                        continue
                    # 查找line中分词w是否在word2id某一key中，如果在，则把tp[word2id[key]]设为1，即表示包含该关键词
                    for key in word2id:
                        if w in key:
                            tp[word2id[key]] += 1
                    # 查找line中分词w是否在word2id_cat某一key中，如果在，则在对于场景命中关键词的位置+1
                    for key in word2id_cat:
                        if w in key:
                            tp[-(word2id_cat[key] + 2)] += 1
                # 该条文本属于哪个场景则tp最后一个位置写几
                tp[-1] = i
                # tp放入data，data为训练文本转化为的文本向量，用于后续的训练模型
                data.append(tp)
        data = np.array(data)
        return data



