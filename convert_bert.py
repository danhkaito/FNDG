import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import utils
from sklearn.metrics import accuracy_score
import os

parser = utils.get_parser()

args = parser.parse_args()
if not os.path.exists(f"../clean data/{args.dataset}/data_embedding/{args.name_model}"):
    os.makedirs(f"../clean data/{args.dataset}/data_embedding/{args.name_model}")
args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Load the dataset into a pandas dataframe.
df = pd.read_csv(f"../clean data/{args.dataset}/train.csv")

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
texts=df['Full text']
label=df['Label']

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))
labels = y[:,0]
print(np.unique(labels))
sentences = texts.values


# Load the BERT tokenizer.
print('Loading BERT tokenizer...')

tokenizer = AutoTokenizer.from_pretrained(args.name_model)
model = AutoModel.from_pretrained(args.name_model)


input_ids = []
attention_masks = []
tokens = dict()
# For every sentence...
for sent in tqdm(sentences):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = args.token_length,         # Pad & truncate all sentences.
                    truncation = True, 
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'][0])
    # print(encoded_dict['input_ids'].shape)
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'][0])
# Convert the lists into tensors.
tokens['input_ids'] = torch.stack(input_ids)
tokens['attention_mask'] = torch.stack(attention_masks)

labels = torch.from_numpy(labels)
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'],  labels)

dataloader = DataLoader(
            dataset, # The validation samples.
            shuffle=False,
            batch_size = args.batch_size # Evaluate with this batch size.
        )
model.to(device)
train_embedding = []
train_labels = []
for batch in tqdm(dataloader):
    # Add batch to GPU
    
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        output = model(b_input_ids, b_input_mask)

    
    embeddings = output.last_hidden_state
    mask = b_input_mask.unsqueeze(-1).expand(embeddings.size()).float()

    mask_embeddings = embeddings * mask
    summed = torch.sum(mask_embeddings, 1)

    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().cpu().numpy()
    train_embedding.append(mean_pooled)
    # Move logits and labels to CPU
    train_labels.append(b_labels.detach().cpu().numpy())


train_embedding = np.concatenate(train_embedding, axis = 0)
train_labels = np.concatenate(train_labels, axis = 0)
train_sentence = sentences

print(train_labels)
print(train_embedding.shape[0], train_labels.shape[0], train_sentence.shape[0])

for i in range(0, len(train_labels)):
    if train_labels[i] != labels[i]:
        print("WRONG")
        print(train_labels[i], labels[i])

with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_embedding_{args.token_length}.npy", "wb") as f:
    np.save(f, train_embedding)
with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_label_{args.token_length}.npy", "wb") as f:
    np.save(f, train_labels)
with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/train_sentence.npy_{args.token_length}", "wb") as f:
    np.save(f, train_sentence)

################################################################################


df = pd.read_csv(f"../clean data/{args.dataset}/test.csv")

# Report the number of sentences.
print('Number of testing sentences: {:,}\n'.format(df.shape[0]))
texts=df['Full text']
label=df['Label']

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))
labels = y[:,0]
print(np.unique(labels))
sentences = texts.values


# Load the BERT tokenizer.
print('Loading BERT tokenizer...')

tokenizer = AutoTokenizer.from_pretrained(args.name_model)
model = AutoModel.from_pretrained(args.name_model)


input_ids = []
attention_masks = []
tokens = dict()
# For every sentence...
for sent in tqdm(sentences):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = args.token_length,         # Pad & truncate all sentences.
                    truncation = True, 
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'][0])
    # print(encoded_dict['input_ids'].shape)
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'][0])

# Convert the lists into tensors.
tokens['input_ids'] = torch.stack(input_ids)
tokens['attention_mask'] = torch.stack(attention_masks)

labels = torch.from_numpy(labels)
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'],  labels)

dataloader = DataLoader(
            dataset, # The validation samples.
            shuffle = False, # Pull out batches sequentially.
            batch_size = args.batch_size # Evaluate with this batch size.
        )
model.to(device)
test_embedding = []
test_labels = []
for batch in tqdm(dataloader):
    # Add batch to GPU
    
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        output = model(b_input_ids, b_input_mask)

    
    embeddings = output.last_hidden_state
    mask = b_input_mask.unsqueeze(-1).expand(embeddings.size()).float()

    mask_embeddings = embeddings * mask
    summed = torch.sum(mask_embeddings, 1)

    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().cpu().numpy()
    test_embedding.append(mean_pooled)
    # Move logits and labels to CPU
    test_labels.append(b_labels.detach().cpu().numpy())


test_embedding = np.concatenate(test_embedding, axis = 0)
test_labels = np.concatenate(test_labels, axis = 0)
test_sentence = sentences

# embedding_list = np.concatenate((train_embedding, test_embedding))
# print(embedding_list.searchsorted)
print(test_embedding.shape[0], test_labels.shape[0], test_sentence.shape[0])

for i in range(0, len(test_labels)):
    if test_labels[i] != labels[i]:
        print("WRONG")
        print(test_labels[i], labels[i])


with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/test_embedding_{args.token_length}.npy", "wb") as f:
    np.save(f, test_embedding)
with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/test_label.npy_{args.token_length}", "wb") as f:
    np.save(f, test_labels)
with open(f"../clean data/{args.dataset}/data_embedding/{args.name_model}/test_sentence_{args.token_length}.npy", "wb") as f:
    np.save(f, test_sentence)
