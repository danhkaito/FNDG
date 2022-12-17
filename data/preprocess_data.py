from dataclasses import replace
from fnmatch import translate
import string
from tkinter import NW
import pymongo
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer
import unicodedata as ud
from pyvi import ViTokenizer
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
import random
"""
    Start section: Get data from database
    Save data to corpus variable
"""

client = pymongo.MongoClient("mongodb+srv://fbfighter:fbfighter@fb-topic.ixbkp2u.mongodb.net/?retryWrites=true&w=majority")
db = client["fakenews"]
col = db["fbpost"]
data = col.find()

corpus = []
labels = []
urls=[]
for x in data:
  if 'is_fakenew' in x:
    corpus.append(x['text'])    
    if x['is_fakenew']!= True and x['is_fakenew']!=False:
        continue
    labels.append(x['is_fakenew'])
    urls.append(x['post_url'])

db = client["fake-news"]
col = db["testpostcmtreactor"]
data = col.find()

for x in data:
  if 'is_fakenew' in x:
    corpus.append(x['text'])    
    labels.append(x['is_fakenew'])
    urls.append(x['post_url'])


""" 
    End section: Get data from database
"""

"""
    Begin section: Global variables
"""
lower_letters = string.ascii_lowercase

text2num = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

maplist = {}
with open('mapping_list.txt', 'r', encoding="utf-8") as f:
    for i in f:
        temp = i.rstrip('\n').split(' ')
        maplist[temp[0]] = temp[1]
        
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
"""
    End section: Global variables
"""

"""
    Begin section:Remove punctualtion
"""
kytudb = ['\u035f', '̼', '\u0320', '‼', '\u0334', '\u0335', '\u05bc', '\u0336', '\u0337']
dict_dau_cau = {
    '\u0311': '\u0302', # ^
    '\u0304': '\u0302', # ^
    '\u0340': '\u0300', # dau huyen
    '\u0341': '\u0301', # dau sac
    '\u0342': '\u0303', # dau nga
    '\u030c': '\u0306'  # dau ă
    
}
def remove_punctualtion(text):
    special_char = kytudb + ['“', '”', '–', '‘', '’', "…","\u0332" ,'\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufe0f']
    punctualtion_list = string.punctuation + "".join(special_char)
    removed_punctuation = "".join([i for i in text if i not in punctualtion_list])
    return removed_punctuation
"""
    End section:Remove punctualtion
"""

"""
    Begin section: Convert dau cau
"""
def convert_dau_cau_va_mapping(text):
    for key in  maplist:
        text = text.replace(key, maplist[key])
        
    for key in dict_dau_cau:
        text = text.replace(key, dict_dau_cau[key])
    return text
"""
    End section: Convert dau cau
"""

# for i in clean_text:
#     print("post-----------")
#     print(i)

"""
    Begin section: Convert to unicode characters
"""
def loaddicchar():
    dict_char = {}
    char_1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    char_utf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    for i in range(len(char_1252)):
        dict_char[char_1252[i]] = char_utf8[i]
    return dict_char


def convert_unicode(txt):
    dict_char = loaddicchar()
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dict_char[x.group()],
        txt,
    )
    
def loaddicchar2():
    dict_char = {}
    char_other_1a = "ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự".split("|")
    char_other_1b = "ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự".split("|")
    for i in range(len(char_other_1a)):
        dict_char[char_other_1a[i]] = char_other_1b[i]
    return dict_char

def convert_unicode2(txt):
    dict_char = loaddicchar2()
    return re.sub(
        r"ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự",
        lambda x: dict_char[x.group()],
        txt,
    )
    
def loaddicchar3():
    dict_char = {}
    char_other_1a = "ặ|ậ|ệ|ộ".split("|")
    char_other_1b = "ặ|ậ|ệ|ộ".split("|")
    for i in range(len(char_other_1a)):
        dict_char[char_other_1a[i]] = char_other_1b[i]
    return dict_char

def convert_unicode3(txt):
    dict_char = loaddicchar3()
    return re.sub(
        r"ặ|ậ|ệ|ộ",
        lambda x: dict_char[x.group()],
        txt,
    )
    
"""
    End section: Convert to unicode characters
"""


"""
    Begin section: Replace special characters with characters in maplist
"""
def find_special_char(text):
    regex = "[^a-z0-9A-Z_\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]"
    set_char = set(re.findall(regex, text))
    return set_char


def replace_special_char(text):
    mapping_list = {}
    set_char = find_special_char(text)
    for char in set_char:
        try:
            char_name = ud.name(char).lower()
            words = char_name.split(' ')
            
            #for number
            if words[-1] in text2num:
                mapping_list[ord(char)] = text2num[words[-1]]
                
            else: # for letter
                for word in words:
                    if word in lower_letters:
                        mapping_list[ord(char)] = word
                        break
        except:
            pass
    return text.translate(str.maketrans(mapping_list))
"""
    End section: Replace special characters with characters in maplist
"""


"""
    Start section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
    Ví dụ: thủy = thuyr, tượng = tuwowngj
"""
bang_nguyen_am = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]
bang_ky_tu_dau = ["", "f", "s", "r", "x", "j"]

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def vn_word_to_telex_type(word):
    dau_cau = 0
    new_word = ""
    for char in word:
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            new_word += char
            continue
        if y != 0:
            dau_cau = y
        new_word += bang_nguyen_am[x][-1]
    new_word += bang_ky_tu_dau[dau_cau]
    return new_word


def vn_sentence_to_telex_type(sentence):
    """
    Chuyển câu tiếng việt có dấu về kiểu gõ telex.
    :param sentence:
    :return:
    """
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = vn_word_to_telex_type(word)
    return " ".join(words)


"""
    End section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
"""


"""
    Start section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
    Xem tại đây: https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF
"""

def standardize_word(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == "i" else bang_nguyen_am[9][dau_cau]
            return "".join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return "".join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return "".join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def standardize_sentence(sentence):
    """
    Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
    :param sentence:
    :return:
    """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1#\2#\3", word).split("#")
        print(cw)
        if len(cw) == 3:
            cw[1] = standardize_word(cw[1])
        words[index] = "".join(cw)
    return " ".join(words)
"""
    End section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
"""


"""
    Start section: Tokenize
    Remove stop word and tokenize
"""
stop_word_list = []

with open('stop_word.txt', 'r', encoding="utf-8") as f:
    for i in f:
        stop_word_list.append(i.rstrip('\n'))

"""
    Important
"""
def preprocess_text(text):
    text = text.lower()
    text = remove_punctualtion(text)
    text = replace_special_char(text)
    text = convert_dau_cau_va_mapping(text)
    
    text = convert_unicode2(text)
    text = convert_unicode(text)
    text = convert_unicode3(text)
    
    set_char = find_special_char(text)
    text = "".join([i for i in text if i not in set_char])
    
    # words = re.split('\s+', text)
    
    # for i, word in enumerate(words):
    #     words[i] = convert_unicode2(word)
        
    # for i, word in enumerate(words):
    #     words[i] = convert_unicode(word)
    
    # for i, word in enumerate(words):
    #     words[i] = convert_unicode3(word)

    # return [word for word in words if word not in stop_word_list]
    text= re.sub('\s+', ' ', text)
    return ViTokenizer.tokenize(text)
"""
    End section: Tokenize
"""

def encode_writting_style(text):
    text = text.lower()
    words = re.split('\s+', text) #list of words
    words = list(filter(None, words))
    num_dot_token = 0
    num_special_token = 0
    
    # test1 = []
    # test2 = []
    if len(words) == 0:
        return [0,0]
    
    for word in words:
        # check dot token
        for i in range(len(word)):
            if i > 0 and i < len(word)-1:
                if word[i-1] != '.' and word[i] == '.' and word[i+1] != '.':
                    num_dot_token += 1
                    # test1 += [word]
                    break
        
        # check special token
        if check_special_char(word):
            num_special_token += 1
            # test2 += [word]
    return [num_dot_token/len(words), num_special_token/len(words)]
"""
    Important
"""

def check_special_char(word):
    for char in word:
        if (char in maplist) or (char in kytudb):
            return True

    set_char = find_special_char(word)
    for char in set_char:
        try:
            char_name = ud.name(char).lower()
            words = char_name.split(' ')
            
            #for number
            if words[-1] in text2num:
                return True
                
            else: # for letter
                for w in words:
                    if w in lower_letters:
                        return True
        except:
            pass
        
    return False



# with open('e.txt', 'w', encoding="utf-8") as f:
#     for i in corpus[:100]:
#         f.write(f"{i}\n")
#         x = encode_writting_style(i)
#         f.write(f"{x}\n\n")
#         # f.write(f"{x[1]}\n")
#         # f.write(f"{x[2]}\n\n")

# with open('a.txt', 'w', encoding="utf-8") as f:
#     for (i, char) in enumerate(find_special_char()):
#         # if i % 20 == 0: f.write('\n')
#         try:
#             f.write(char + "\t" + ud.name(char) + "\n")
#         except:
#             pass
        
# with open('c.txt', 'w', encoding="utf-8") as f:
#     set_char = set()
#     for text in clean_text2:
#         set_char.update(find_special_char(text))
        
#     for (i, char) in enumerate(set_char):
#         try:
#             f.write(char + " " + ud.name(char) + "\n")
#         except:
#             pass



# original_stdout = sys.stdout
# with open('a.txt', 'w') as f:
#     sys.stdout = f
#     print(find_special_char())
#     sys.stdout = original_stdout

# test_text = list(clean_text)[2]

# token_text_list = list(map(tokenization, clean_text2))

texts_list = list(map(preprocess_text, corpus))
writting_style = list(map(encode_writting_style, corpus))
# print(texts_list[1])

# for i in token_text_list[:20]:
#   print(i)

labeled_data = list(zip(corpus, texts_list, writting_style, urls, labels))

with open('../../dataset/data_label_all', 'wb') as f:
    pickle.dump(labeled_data, f) 


# import csv
  
  
# # field names 
# fields = ['Văn bản gốc', 'Văn bản xử lý', 'Writing Style', 'URLs', 'Labels'] 
    
# # data rows of csv file 

# import xlsxwriter
 
# workbook = xlsxwriter.Workbook('../../dataset/data_fakenews.xlsx')
 
# # By default worksheet names in the spreadsheet will be
# # Sheet1, Sheet2 etc., but we can also specify a name.
# worksheet = workbook.add_worksheet("Data_Imbalance")
 
# # Some data we want to write to the worksheet.

 
# # Start from the first cell. Rows and
# # columns are zero indexed.
# row = 0
# col = 0
# for i in range(0, len(fields)):
#     worksheet.write(row, col+i, fields[i])
# row+=1
# # Iterate over the data and write it out row by row.
# for data_each in labeled_data:
#     for i in range(0, len(data_each)):
#         worksheet.write(row, col+i, str(data_each[i]))
#     row += 1
 


################## TRAIN TEST SPLIT BALANCE #################
# data_unlabeled = [x[:-1] for x in labeled_data]

# print(data_unlabeled[1])
# print("Done Preprocess")

# labels = np.array([x[-1] for x in labeled_data])
# fake_idx = np.squeeze(np.argwhere(labels == True))
# X_fake_train, X_fake_test = train_test_split(fake_idx, test_size=0.2, random_state=42)
# print(fake_idx)
# true_idx = np.squeeze(np.argwhere(labels == False))
# print(true_idx)
# X_true_train, X_true_test = train_test_split(true_idx, test_size=1-(len(fake_idx)+50)/len(true_idx), random_state=42)

# train_data_idx =  np.concatenate((X_fake_train, X_true_train), axis=None)
# test_data_idx = np.concatenate((X_fake_test, X_true_test), axis=None)

# print(len(fake_idx))
# print(fake_idx, X_true_train)
# data_unlabeled_equal = [labeled_data[i] for i in train_data_idx]
# data_unlabeled_equal_test = [labeled_data[i] for i in test_data_idx]
# print(len(data_unlabeled_equal))

# with open('../../dataset/data_preprocess_balance', 'wb') as f:
#     pickle.dump(data_unlabeled_equal, f)

# with open('../../dataset/data_preprocess_balance_test', 'wb') as f:
#     pickle.dump(data_unlabeled_equal_test, f)

################################################################

################# TRAIN TEST SPLIT IMBALLANCE ##################
data_train, data_test = train_test_split(labeled_data, test_size=0.2, random_state=42)

with open('../../dataset/data_preprocess_imbalance_train', 'wb') as f:
    print(len(data_train))
    labels = np.array([x[-1] for x in data_train])
    print(len(np.squeeze(np.argwhere(labels == True))))
    print(len(np.squeeze(np.argwhere(labels == False))))

    pickle.dump(data_train, f)

with open('../../dataset/data_preprocess_imbalance_test', 'wb') as f:
    print(len(data_test))
    labels = np.array([x[-1] for x in data_test])
    print(len(np.squeeze(np.argwhere(labels == True))))
    print(len(np.squeeze(np.argwhere(labels == False))))

    pickle.dump(data_test, f)





############# MISC ##############################
# worksheet = workbook.add_worksheet("Data_Balance")
 
# # Some data we want to write to the worksheet.

 
# # Start from the first cell. Rows and
# # columns are zero indexed.
# row = 0
# col = 0
# for i in range(0, len(fields)):
#     worksheet.write(row, col+i, fields[i])
# row+=1
# # Iterate over the data and write it out row by row.
# for data_each in data_unlabeled_equal:
#     for i in range(0, len(data_each)):
#         worksheet.write(row, col+i, str(data_each[i]))
#     row += 1
 
# workbook.close()




# token_text_list = list(filter(None, token_text_list))

# token_text_list2 = []

# for i in token_text_list:
#     newarr = []
#     for token in i:
#         set_char = find_special_char(token)
#         newarr.append("".join([k for k in token if k not in set_char]))
#     token_text_list2.append(newarr)
    

# test = {}

# for id1,list_words in enumerate(token_text_list):
#         for id2, word in enumerate(list_words):
#             set_token = find_special_char(word)
#             if len(set_token) > 0:
#                 test[word] = token_text_list2[id1][id2]

# with open('c.txt', 'w', encoding="utf-8") as f:
#     for i in range(1000):
#         f.write(corpus[i]+'\n----------------\n')
#         f.write(token_text_list[i] + '\n=======================\n\n')
        

# for i in token_text_list[:20]:
#     print(i)