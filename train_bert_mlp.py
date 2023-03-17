import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from  model.model import BertClassifier
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from early_stoping import EarlyStopping

parser = utils.get_parser()

args = parser.parse_args()
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
tokenizer = BertTokenizer.from_pretrained(args.name_model)

input_ids = []
attention_masks = []
tokens = dict()
# For every sentence...
for sent in sentences:
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


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'],  labels)

# Create a 80-20 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = args.batch_size

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
model = BertClassifier(args.name_model, args.num_class, args.dropout)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr= args.lr, eps=args.eps)
total_steps = len(train_dataloader) * args.epoch

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = 0, # Default value in run_glue.py
                                    num_training_steps = total_steps)

model = model.to(device)
criterion = criterion.to(device)

early_stopping = EarlyStopping(patience=args.patience)

def train(epochs):
    training_stats = []
    
    for epoch_num in range(epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_num + 1, epochs))
        print('Training...')


        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for train_input_ids, train_input_mask, train_label in tqdm(train_dataloader):
            train_input_ids = train_input_ids.to(device)
            train_input_mask = train_input_mask.to(device)
            train_label = train_label.to(device)

            output = model(train_input_ids, train_input_mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input_ids, val_input_mask, val_label in validation_dataloader:

                val_input_ids = val_input_ids.to(device)
                val_input_mask = val_input_mask.to(device)
                val_label = val_label.to(device)

                output = model(val_input_ids, val_input_mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        # Early stoping
        if args.early_stopping:
            if early_stopping.early_stop(total_loss_val, model, "BERT_" + args.dataset, args.token_length):
                print(f"Early stopping at epoch: {epoch_num + 1}")
                break
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / train_size: .3f} \
            | Train Accuracy: {total_acc_train / train_size: .3f} \
            | Val Loss: {total_loss_val / val_size: .3f} \
            | Val Accuracy: {total_acc_val / val_size: .3f}')
        
        training_stats.append(
        {
        'epoch': epoch_num + 1,
        'Training Loss': total_loss_train / train_size,
        'Valid. Loss': total_loss_val / val_size,
        'Valid. Accur.': total_acc_train / train_size,
        # 'Training Time': training_time,
        # 'Validation Time': validation_time
        })
        
    return training_stats

training_stats = train(args.epoch)


# Display floats with two decimal places.
# pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
# df_stats = pd.DataFrame(data=training_stats)

# # Use the 'epoch' as the row index.
# df_stats = df_stats.set_index('epoch')

# # A hack to force the column headers to wrap.
# #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# # Display the table.

# # Use plot styling from seaborn.
# sns.set(style='darkgrid')

# # Increase the plot size and font size.
# sns.set(font_scale=1.5)
# plt.rcParams["figure.figsize"] = (12,6)

# # Plot the learning curve.
# plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
# plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# # Label the plot.
# plt.title("Training & Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.xticks([1, 2, 3, 4, 5])

# plt.show()