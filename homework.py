import os
import math
import jieba
import logging
import numpy as np
import matplotlib as mpl

def read_every_files(path):
    data_list = []
    for root, route, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            with open(filename, 'r', encoding='ANSI') as f:
                txt = f.read()
                d = ChineseData()
                d.txt = txt
                d.txtname = file.split('.')[0]
                data_list.append(d)
            f.close()
    return data_list


def get_punctuation_list(path):
    punctuation = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    punctuation.extend(['\n', '\u3000', '\u0020', '\u00A0'])
    return punctuation


def get_stopwords_list(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

class ChineseData:
    def __init__(self, txtname='', txt='', sentences=[], words=[], entropyinit={}):
        self.txtname = txtname
        self.txt = txt
        self.sentences = sentences
        self.words = words
        global punctuation
        self.punctuation = punctuation
        global stopwords
        self.stopwords = stopwords
        self.entropy = entropyinit

    def sepSentences(self):
        line = ''
        sentences = []
        for w in self.txt:
            if w in self.punctuation and line != '\n':
                if line.strip() != '':
                    sentences.append(line.strip())
                    line = ''
            elif w not in self.punctuation:
                line += w
        self.sentences = sentences

    def sepWords(self):
        words = []
        dete_stopwords = 0
        if dete_stopwords:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i]) if x not in self.stopwords])
        else:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i])])
        self.words = words

    def getNmodel(self, phrase_model, n):
        if n == 1:
            for i in range(len(self.words)):
                phrase_model[self.words[i]] = phrase_model.get(self.words[i], 0) + 1
        else:
            for i in range(len(self.words) - (n - 1)):
                if n == 2:
                    condition_t = self.words[i]
                else:
                    condition = []
                    for j in range(n - 1):
                        condition.append(self.words[i + j])
                    condition_t = tuple(condition)
                phrase_model[(condition_t, self.words[i + n - 1])] = phrase_model.get(
                    (condition_t, self.words[i + n - 1]), 0) + 1
                
    def getN_1model(self, phrase_model, n):
        if n == 1:
            for i in range(len(self.words)):
                phrase_model[self.words[i]] = phrase_model.get(self.words[i], 0) + 1
        else:
            for i in range(len(self.words) - (n - 1)):
                condition = []
                for j in range(n):
                    condition.append(self.words[i + j])
                condition_t = tuple(condition)
                phrase_model[condition_t] = phrase_model.get(condition_t, 0) + 1

    def calcuNmodelEntropy(self, n, entropy_dic):
        if n < 1 or n >= len(self.words):
            print("Wrong N!")
        elif n == 1:
            phrase_model = {}
            self.getNmodel(phrase_model, 1)
            model_lenth = len(self.words)
            entropy_dic[n] = sum(
                [-(phrase[1] / model_lenth) * math.log(phrase[1] / model_lenth, 2) for phrase in phrase_model.items()])
            entropy_dic[n] = round(entropy_dic[n], 4)
        else:
            phrase_model_n = {}
            phrase_model_n_1 = {}
            self.getNmodel(phrase_model_n, n)
            self.getN_1model(phrase_model_n_1, n - 1)
            phrase_n_len = sum([phrase[1] for phrase in phrase_model_n.items()])
            entropy = []
            for n_phrase in phrase_model_n.items():
                p_xy = n_phrase[1] / phrase_n_len
                p_x_y = n_phrase[1] / phrase_model_n_1[n_phrase[0][0]]
                entropy.append(-p_xy * math.log(p_x_y, 2))
            entropy_dic[n] = round(sum(entropy), 4)
    
    def run(self):
        self.sepSentences()
        self.sepWords()
        entropy_dic = {}
        self.calcuNmodelEntropy(1, entropy_dic)
        self.calcuNmodelEntropy(2, entropy_dic)
        self.calcuNmodelEntropy(3, entropy_dic)
        self.entropy = entropy_dic

if __name__ == "__main__":
    data_dir_path = './/小说全集'
    stopwords_path = './/cn_stopwords.txt'
    punctuation_path = './/cn_punctuation.txt'

    global stopwords
    stopwords = get_stopwords_list(stopwords_path)
    global punctuation
    punctuation = get_punctuation_list(punctuation_path)
    data_list = read_every_files(data_dir_path)

    for i in range(len(data_list)):
        logging.info('处理《' + data_list[i].txtname + '》...')
        ChineseData.run(data_list[i])
        logging.info('《' + data_list[i].txtname + '》完成...')

    np.save('data_list.npy', data_list)
    print("数据已保存")