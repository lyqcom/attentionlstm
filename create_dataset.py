import os
import numpy as np

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

class Sentence(object):
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target
        self.solution = np.zeros(grained, dtype=np.float32)
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating)+1] = 1
        except:
            exit()

    def stat(self, target_dict, wordlist, grained=3):
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=np.float32)
        for word in self.content.split(' '):
            data.append(wordlist[word])
            try:
                pol = Lexicons_dict[word]
                solution[i][pol+1] = 1
            except:
                pass
            i = i+1
        for word in self.target.split(' '):
            data_target.append(wordlist[word])
        return {'seqs': data, 
                'target': data_target, 
                'solution': np.array([self.solution]), 
                'target_index': self.get_target(target_dict)}

    def get_target(self, dict_target):
        return dict_target[self.target]

class DataManager(object):
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        for fname in self.fileList:
            data = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in range(int(len(sentences)/3)):
                    content, target, rating = sentences[i*3].strip(), sentences[i*3+1].strip(), sentences[i*3+2].strip()
                    sentence = Sentence(content, target, rating, grained)
                    data.append(sentence)
            self.origin[fname] = data
        self.gen_target()

    def gen_word(self):
        wordcount = {}
        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        words = wordcount.items()
        sorted(words, key=lambda x:x[1], reverse=True)
        self.wordlist = {item[0]:index+1 for index, item in enumerate(words)}
        return self.wordlist

    def gen_target(self, threshold=5):
        self.dict_target = {}
        for fname in self.fileList:
            for sent in self.origin[fname]:
                if sent.target in self.dict_target:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        for (key,val) in self.dict_target.items():
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target

    def gen_data(self, grained=3):
        self.data = {}
        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))


def _convert_to_mindrecord(mindrecord_path, data):
    """
    convert cor dataset to mindrecord dataset
    """
    print("convert to mindrecord...")

    content = []
    sen_len = []
    aspect = []
    solution = []
    aspect_index = []

    for info in data:
        content.append(info['seqs'])
        aspect.append(info['target'])
        sen_len.append([len(info['seqs'])])
        solution.append(info['solution'])
        aspect_index.append(info['target_index'])

    padded_content = np.zeros([len(content), 50])
    for index, seq in enumerate(content):
        if len(seq) <= 50:
            padded_content[index, 0:len(seq)] = seq
        else:
            padded_content[index] = seq[0:50]

    content = padded_content
    
    if os.path.exists(mindrecord_path):
        os.remove(mindrecord_path)
        os.remove(mindrecord_path + ".db")

    # schema
    schema_json = {"content": {"type": "int32", "shape": [-1]},
                   "sen_len": {"type": "int32"},
                   "aspect": {"type": "int32"},
                   "solution": {"type": "int32", "shape": [-1]}}

    data_list = []
    for i in range(len(content)):
        sample = {"content": content[i],
                   "sen_len": int(sen_len[i][0]),
                   "aspect": int(aspect[i][0]),
                   "solution": solution[i][0]}
        data_list.append(sample)

    writer = FileWriter(mindrecord_path, shard_num=1)
    writer.add_schema(schema_json, "lstm_schema")
    writer.write_raw_data(data_list)
    writer.commit()

def wordlist_to_glove_weight(wordlist, glove_file):
    glove_word_dict = {}
    with open(glove_file) as f:
        line = f.readline()
        while line:
            array = line.split(' ')
            word = array[0]
            glove_word_dict[word] = array[1:301]
            line = f.readline()
    print(glove_word_dict['food'])
    print(len(glove_word_dict['food']))
    
    weight = np.zeros((len(wordlist), 300)).astype(np.float32)+0.01
    unfound_count = 0
    for word, i in wordlist.items():
        word = word.strip()
        if word in glove_word_dict:
            weight[i-1] = glove_word_dict[word]
        else:
            unfound_count += 1
    
    print("not found in glove: ", unfound_count)
    print(np.shape(weight))
    print(weight.dtype)

    np.savez('weight.npz', weight)



if __name__ == "__main__":

    data = DataManager('/root/xidian_wks/lhh/new_ATAE/data')
    wordlist = data.gen_word()
    # print("wordlist: ", type(wordlist))

    # wordlist_to_glove_weight(wordlist, '/root/xidian_wks/lhh/glove_file/glove.840B.300d.txt')

    train_data, dev_data, test_data = data.gen_data(grained=3)

    _convert_to_mindrecord('./mine_dataset/train.mindrecord', train_data)
    _convert_to_mindrecord('./mine_dataset/test.mindrecord', test_data)

    # print("content: ", np.shape(padded_content))
    # print("sen_len: ", np.shape(sen_len))
    # print("aspect: ", np.shape(aspect))
    # print("solution: ", np.shape(solution))
    # print("aspect_index: ", np.shape(aspect_index))

    # print("content: ", content[1])
    # print("sen_len: ", sen_len[1])
    # print("aspect: ", aspect[1])
    # print("solution: ", solution[1][0])
    # print("aspect_index: ", aspect_index[1])



