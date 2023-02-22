from model.KSOM.PNode import *
from model.KSOM.CNode import *
from model.KSOM.CSom import *

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle 
# from utils import *
# import unicodedata as ud
# import scipy
# from sklearn.model_selection import train_test_split

METHOD = 'euclid'   # 'euclid' or 'cosin'
NUM_ITER = 50000
RADIUS_MAP = 20
# Read data
# train_df = pd.read_csv('benchmark_data\\Liar\\train.csv')
# test_df = pd.read_csv('benchmark_data\\Liar\\test.csv')

# train_X = train_df['Statement'].values
# train_Y = train_df['Label'].values

# test_X = train_df['Statement'].values
# test_Y = train_df['Label'].values

train_X = np.load('./model_save/Fake_or_Real/data numpy/train_sentence.npy', allow_pickle=True)
train_Y = np.array(np.load('./model_save/Fake_or_Real/data numpy/train_label.npy', allow_pickle=True))
convert_label = lambda t: (t == 'FAKE').astype(int)
train_Y = convert_label(train_Y)

# print(len(data_train))
doc_2_vec = np.load('./model_save/Fake_or_Real/data numpy/train_embed.npy')
# doc_2_vec = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1,1), stop_words = 'english')
model = CSom(RADIUS_MAP, train_X, train_Y, NUM_ITER, doc_2_vec)
model.Train(METHOD)
model.map_PNode2CNode(METHOD)

# PNodes = TfidfVectorizer()
# PNodes = PNodes.fit_transform(corpus_val).todense()

# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(corpus_val[i], PNodes[i])

print("Saving Model...")
ksom_Weights=open('./model/KSOM/ksom_model_100k_euclid_idf.ckpt', 'wb')
model.save(ksom_Weights)
print("Finish Saving Model")

# PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
# PNodes = PNodes.fit_transform(token_text_list).todense()
# sum_quan_err=0
# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     sum_quan_err+=math.sqrt(SuitNode.CalculateDistance(np.squeeze(np.asarray(PNodes[i]))))
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(token_text_list[i], PNodes[i])
# print(f"Quantization Error {sum_quan_err/len(token_text_list)}")
# for iy, ix in np.ndindex(model.m_Som.shape):
#     for i in range(len(model.m_Som[iy,ix].PNodes)):
#         print(iy," ",ix," ",model.m_Som[iy,ix].PNodes[i].corpus)
# model.Plot()