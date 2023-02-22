from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
train_df = pd.read_csv('benchmark_data\\Liar\\train.csv')
test_df = pd.read_csv('benchmark_data\\Liar\\test.csv')

train_X = train_df['Statement'].values
train_Y = train_df['Label'].values

test_X = train_df['Statement'].values
test_Y = train_df['Label'].values





doc_2_vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range = (1,1), stop_words = 'english', max_features=3000)

vecs_train_idf = np.asarray(doc_2_vec.fit_transform(train_X).todense())

vecs_test_idf = np.asarray(doc_2_vec.transform(test_X).todense())
print(vecs_train_idf.shape)
print(len(vecs_train_idf[0]))
np.save('./model_save/embeddings_idf_train.npy', vecs_train_idf)
np.save('./model_save/embeddings_idf_test.npy', vecs_test_idf)


# print(len(train_X))
# print(vecs_train_idf.todense())
# vecs_bert = np.load('./model_save/embeddings.npy')
# print(vecs_bert)