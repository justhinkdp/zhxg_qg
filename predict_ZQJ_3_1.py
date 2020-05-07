# -*- coding:utf-8 -*-
import lightgbm as lgb
import jieba
import re
import numpy as np
path = './data/'


# sentence是词典，存储要预测是语句
def key_cv(sentence):
    word2id={}
    word2id_cat={}
    word2id_cat_m = {}

    # mergerate=0.05

    # tfidf keyword
    for line in open(path+"keywords_single_250.txt", encoding='UTF-8'):
        for w in line.split():
            word2id[w.strip()] = len(word2id)
    ct = 0
    counts = [0,0,0,0,0]

    for line in open(path+"keywords_single_250.txt", encoding='UTF-8'):
        for w in line.split():
            word2id_cat[w.strip()]=ct
            counts[ct] += 1
        ct += 1

    data=[]
    # t1 = open('D:\CodeProject\PythonProject\\nlp_zhxg\\'+testfile+'.txt')  # 打开头目录的要预测的文件，因为是第一次打开，还没有预测后剩下的other文件
    # d1 = t1.read().split('\r')

    # print len(d1)
    # t1.close()
    d1 = sentence
    kdr = []

    for s in d1:
        if d1[s] == '':
            continue
        #content = d1[s].split('|')[2] # s是key，content是需要预测的语句
        content = d1[s]
        m = re.findall('[\d]+\.wav[\d|！|_|。]+', content)
        for mm in m:
            content = content.replace(mm, '')
        content = re.sub('[\d]+_[\d]+_', '', content)
        tp = [0]*(len(word2id) + 5)

        # 四个set表示四个类别中有哪些关键词在这个语句中命中
        kd = [set(), set(), set(), set(), set()]


        # kdr.append(kd)

        for w in jieba.cut(content):
            for key in word2id:
                if w in key:
                    tp[word2id[key]] += 1
            for key in word2id_cat:
                if w in key:
                    tp[-(word2id_cat[key] + 1)] += 1

        data.append(tp)
        kdr.append(kd)

    # 处理后得到数组data进行预测
    data=np.array(data)
    # print data.shape
    r=[]
    for i in range(5):
        clf = lgb.Booster(model_file="./models/key_cv" + str(i) + ".m")
        if len(r)==0:
            r=clf.predict(data)
        else:
            r+=clf.predict(data)

    rr = ['2', '1', '0', '-1', '-2']
    tow = open('result.txt', 'w',encoding='UTF-8')
    # townext = open(path + testfile+'_others.txt', 'wb')

    # 将预测结果r与原始数据文字部分d1打包，即r与d1一一对应,d1为词典，在这里v[1]是词典的key
    for v in list(zip(r,d1,kdr)):
        tpr=np.where(v[0][:]==max(v[0][:]))[0][0]
        # print(tpr)
        b = np.argsort(np.array(list(v[0])))
        # print(v[0])
        # print(rr[tpr])
        value = rr[tpr]
        write_str = str(v[1])+':'+ value + "\n"
        tow.write(write_str)
        # 直接输出，输出为：种类+语句
        # print rr[tpr] + "|" + str(round(v[0][tpr], 2)) + "|" + v[1].strip() + d1[v[1]]
        # 删掉预测过的语句

        
    tow.flush()
    tow.close()
    
    print("预测完成")
