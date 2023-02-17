from model.KSOM.PNode import *
from model.KSOM.CNode import *
from model.KSOM.CSom import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle 
# from utils import *
# import unicodedata as ud
# import scipy
# from sklearn.model_selection import train_test_split


# Read data
train_df = pd.read_csv('benchmark_data\\Liar\\train.csv')
test_df = pd.read_csv('benchmark_data\\Liar\\test.csv')

train_X = train_df['Statement'].values
train_Y = train_df['Label'].values

test_X = train_df['Statement'].values
test_Y = train_df['Label'].values


# print(len(data_train))
doc_2_vec = np.load('./model_save/embeddings.npy')
# doc_2_vec = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1,1), stop_words = 'english')
model = CSom(30, train_X, 40000, doc_2_vec)
model.Train()
# PNodes = TfidfVectorizer()
# PNodes = PNodes.fit_transform(corpus_val).todense()

# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(corpus_val[i], PNodes[i])

print("Saving Model...")
ksom_Weights=open('./model/KSOM/ksom_model.ckpt', 'wb')
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