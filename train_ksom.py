from model.KSOM.PNode import *
from model.KSOM.CNode import *
from model.KSOM.CSom import *
import utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import os
# import pickle 
# from utils import *
# import unicodedata as ud
# import scipy
# from sklearn.model_selection import train_test_split

parser = utils.get_parser()

args = parser.parse_args()

METHOD = args.method_ksom   # 'euclid' or 'cosin'
# Read data
# train_df = pd.read_csv('benchmark_data\\Liar\\train.csv')
# test_df = pd.read_csv('benchmark_data\\Liar\\test.csv')

# train_X = train_df['Statement'].values
# train_Y = train_df['Label'].values

# test_X = train_df['Statement'].values
# test_Y = train_df['Label'].values

train_sentence = np.load(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_sentence_{args.token_length}.npy", allow_pickle=True)
train_Y = np.load(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_label_{args.token_length}.npy")
train_X = np.load(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_embedding_{args.token_length}.npy")
NUM_ITER = args.epoch*train_X.shape[0]

RADIUS_MAP = int(math.sqrt((5*math.sqrt(train_X.shape[0]))))
# print(train_sentence)
# print(len(data_train))
# train_X = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1,1), stop_words = 'english')
model = CSom(RADIUS_MAP,train_X, train_Y, train_sentence, NUM_ITER)
model.Train(METHOD)

# PNodes = TfidfVectorizer()
# PNodes = PNodes.fit_transform(corpus_val).todense()

# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(corpus_val[i], PNodes[i])

print("Saving Model...")
if not os.path.exists(f'./model/KSOM/{args.dataset}/{args.name_model}'):
    os.makedirs(f'./model/KSOM/{args.dataset}/{args.name_model}')

ksom_Weights=open(f'./model/KSOM/{args.dataset}/{args.name_model}/{args.token_length}_ksom_{args.method_ksom}.ckpt', 'wb')
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