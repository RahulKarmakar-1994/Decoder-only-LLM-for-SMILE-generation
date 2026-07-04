

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:26:51 2024

@author: rahulkarmakar
"""

[ord(x)for x in "안녕하세요 👋 (hello in Korean!)"]

a=[ord(x)for x in "*CC(*)FE=O"]


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition

import torch
# In[2]:


from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
SMILES_CHARS=['.','c', 'n', 'o', 'C', 'N', 'F', 'K','=','O', 'H', '%','0',']', '[', 'Na','Li', 'Ca','Cd','Te','(', ')', '1','\ ','\\','2','#','Cl','/','B','s','S','Se','Ge','Br','Sn','Zn','Si','se','I', 'Pb','3', '4', '5', '6', '7', '8', '+','-', '9', 'P','*']

len(SMILES_CHARS)
smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))
# In[3]:


data=pd.read_csv("/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/tg_raw.csv")
data

text = data['SMILES']
#%%
print(text[:1000])

#%%
chars = list(set(text))

print(''.join(chars))

#%%
stoi = {s:i for i,s in enumerate(SMILES_CHARS)}

#stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(vocab_size)   

#%%
X_tot = np.zeros((7174,314,52))
extracted_characters = [[] for _ in range(7174)]
extracted_token = [[] for _ in range(7174)]
j = 0
for sm in range(7174):
    # Initialize an empty list to hold the extracted characters
    #print('hi')
    
    
    # Use a variable to track the index while looping through the string
    i = 0
    
    smiles_string = list(text[sm])
    str_length = len(text[sm])
    # extracted_characters[j].append('.')
    # extracted_token[j].append(0)
    while i < str_length:
        # Check for multi-character elements
        check = "".join(list(smiles_string[i:i+2]))
        if i < str_length-1 and  check in SMILES_CHARS:
            ix = stoi[check]
            extracted_characters[j].append(check)
            extracted_token[j].append(ix)
            #if check in smi2index:
                #X_tot[sm,j,smi2index[check]] = 1
            i += 2  # Skip the next character since we've just added two    
        else:
            if smiles_string[i] =='%':
                print('yes',sm)
            ix =   stoi[smiles_string[i]]  
            extracted_characters[j].append(smiles_string[i])  # Add the single character
            extracted_token[j].append(ix)
             # Move to the next character
            #print(sm,i,smi2index[smiles_string[i]],smiles_string[i])
            #X_tot[sm,j,smi2index[smiles_string[i]]] = 1
            i += 1 
      
    # extracted_characters[j].append('.')   
    # extracted_token[j].append(0)
    j = j +1 
    
    
    
#%%
stoi = {s:i for i,s in enumerate(SMILES_CHARS)}

#stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(vocab_size)   


#%%

smile_generate = text[2713]
from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole
molecule = Chem.MolFromSmiles(smile_generate)
molecule 


#%%

b={}
counts_smile = {}
counts_token = {}
for w in extracted_characters:
    # for ch1,ch2 in zip(w,w[1:]):
    #     biagram = (ch1, ch2)
    #     b[biagram] = b.get(biagram,0)+1
    #     print(ch1,ch2)
    
    for ch1,ch2 in zip(w, w[1:]): # Pythonic way to iterate consecutive elements
        pair = (ch1, ch2)
        counts_smile[pair] = counts_smile.get(pair, 0) + 1
        #print(ch1,ch2,counts[pair])
        
for w in extracted_token:
    # for ch1,ch2 in zip(w,w[1:]):
    #     biagram = (ch1, ch2)
    #     b[biagram] = b.get(biagram,0)+1
    #     print(ch1,ch2)
    
    for ch1,ch2 in zip(w, w[1:]): # Pythonic way to iterate consecutive elements
        pair = (ch1, ch2)
        counts_token[pair] = counts_token.get(pair, 0) + 1
        #print(ch1,ch2,counts[pair])        

#%%

top_pair_smile = max(counts_smile, key=counts_smile.get)
top_pair_smile
#%%
top_pair_token = max(counts_token, key=counts_token.get)
top_pair_token

#%%
new_token = [[] for _ in range(7174)]
def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))

for i in range(len(extracted_token)):
    new_token[i] = merge(extracted_token[i], top_pair_token, 53)
#print(new_token)
print("length:", len(new_token))

#%%  Generate most occuring pair and merge

def get_stats(ids):
    counts = {}
    for w in ids:
        # for ch1,ch2 in zip(w,w[1:]):
            for bgpair in zip(w, w[1:]):
                counts[bgpair] = counts.get(bgpair, 0) + 1
    return counts

def merge_cons(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


# ---
vocab_size = 52 # the desired final vocabulary size
num_merges = vocab_size - 52
ids = list(extracted_token) # copy so we don't destroy the original list
new_smiles = list(SMILES_CHARS)
#%%
merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 52 + i
  print(f"merging {pair} into a new token {idx}")
  for j in range(len(ids)):
      ids[j] = merge_cons(ids[j], pair, idx)
  merges[pair] = idx
  
#%%

for key,value in merges.items():
    out = []
    print(key[0],key[1],value)
    ix = itos[key[0]]
    iy = itos[key[1]]
    out.append(ix)
    out.append(iy)
    print(ix,iy , ''.join(out))
    new_smiles.append(''.join(out))
    itos[value] = ''.join(out)
    ns = ''.join(out)
    stoi[ns] = value
    
#%%    
length = []  
for i in range(len(ids)):
    length.append(len(ids[i]))  
#%%

ids_mod = [[] for _ in range(7174)]
for sm in range(7174):
    j = 0
    str_length = len(ids[sm])
    ids_mod[sm].append(0)
    while j < str_length:
        ix = ids[sm][j]
        ids_mod[sm].append(ix)
        j = j +1 
    # extracted_characters[j].append('.')   
    ids_mod[sm].append(0)
    
#%%

# build the dataset
block_size = 12 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  #print(words)
  for w in words:
    context = [0] * block_size
    for ch in w :
      print(context,ch)
      ix = ch
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
# random.seed(42)
# random.shuffle(ids_mod)

# Generate the same random order
random.seed(42)
indices = list(range(len(ids_mod)))
random.shuffle(indices)

# Shuffle both lists using the same indices
ids_mod = [ids_mod[i] for i in indices]
text_shuffle = [text[i] for i in indices]

n1 = int(0.8*len(ids_mod))
n2 = int(0.9*len(ids_mod))
Xtr,  Ytr  = build_dataset(ids_mod[:n1])     # 80%
Xdev, Ydev = build_dataset(ids_mod[n1:n2])   # 10%
Xte,  Yte  = build_dataset(ids_mod[n2:])     # 10%     

batch_size = 32
from torch.utils.data import TensorDataset, DataLoader
# Create TensorDataset and DataLoader
dataset = TensorDataset(Xtr, Ytr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%%

# Near copy paste of the layers we have developed in Part 3
n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden,vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
parameters = [C, W1, b1, W2, b2]

#%%

sum(p.nelement() for p in parameters) # number of parameters in total

#%%

for p in parameters:
  p.requires_grad = True

#%%
# learning_rate = 0.001
# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
  
  #%%%
max_steps = 20000
batch_size = 32
lossi = []
eval_interval = 10
learning_rate = 0.01
max_iters = 100
start = 0
  
lri = []
lossi = []
stepi = []  
import torch.nn.functional as F 


for i in range(start,max_iters):
    for xb, yb in dataloader:  # This fetches a batch of data
        # Forward pass
        emb = C[xb]  # (batch_size, 3, 2)
        h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, yb)

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        for p in parameters:
          p.data += -learning_rate  * p.grad

        # track stats
        #lri.append(lre[i])
        # track stats
    if i % eval_interval == 0 or i == max_iters - 1:
      print(f'{i:7d}/{max_iters:7d}: {loss.item():.4f}')
    stepi.append(i)
    lossi.append(loss.log10().item())

# for i in range(200000):
  
#   # minibatch construct
#   ix = torch.randint(0, Xtr.shape[0], (32,))
  
#   # forward pass
#   emb = C[Xtr[ix]] # (32, 3, 2)
#   h = torch.tanh(emb.view(-1, 80) @ W1 + b1) # (32, 100)
#   logits = h @ W2 + b2 # (32, 27)
#   loss = F.cross_entropy(logits, Ytr[ix])
#   #print(loss.item())
  
#   # backward pass
#   for p in parameters:
#     p.grad = None
#   loss.backward()
  
#   # update
#   #lr = lrs[i]
#   lr = 0.1 if i < 100000 else 0.01
#   for p in parameters:
#     p.data += -lr * p.grad

#   # track stats
#   #lri.append(lre[i])
#   stepi.append(i)
#   lossi.append(loss.log10().item())
#%%
print(loss.item())
#%%

plt.plot(stepi, lossi)
plt.show() 
  
#%%

# training loss 
emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
loss
#%%

# validation loss
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
loss

#%%
# test loss
emb = C[Xte] # (32, 3, 2)
h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Yte)
loss
#%%
for p in parameters:
    print(f'Weight norm: {p.data.norm():.4f}, Gradient norm: {p.grad.norm():.4f}')


#%%

#torch.save(model, "byte_pair_wavenet_model_vocab_52.pth")
#%%
# put layers into eval mode (needed for batchnorm especially)
# for layer in model.layers:
#   layer.training = False
  
  
#%%  
  # evaluate the loss
# @torch.no_grad() # this decorator disables gradient tracking inside pytorch
# def split_loss(split):
#   x,y = {
#     'train': (Xtr, Ytr),
#     'val': (Xdev, Ydev),
#     'test': (Xte, Yte),
#   }[split]
#   logits = model(x)
#   loss = F.cross_entropy(logits, y)
#   print(split, loss.item())

# #split_loss('train')
# split_loss('test')
#%%
loss_test = (2.25+2.79)/2

#%%  
valid_smile =[]
#%%
max_new_tokens=max(length)
generation_cycle = 20000


for sm in range(generation_cycle):
    
    out = []
    context = [0] * block_size # initialize with all ...
    ix_prev = 0
    for _ in range(max_new_tokens):
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      #print(ix,probs.shape)
      context = context[1:] + [ix]
      if ix_prev!=0 and ix == 0:
        break
      ix_prev = ix
      #if ix < 53:
      if ix !=0 :    
          out.append(ix)
    #out.pop()   
    #print(''.join(out)) 
    #print(out)
    print(sm)
    #print(''.join(itos[i] for i in out))
    sms = ''.join(itos[i] for i in out)
    if sms:
        molecule = Chem.MolFromSmiles(sms)
        if molecule:
            valid_smile.append(''.join(itos[i] for i in out))


# sample from the model
# for sm in range(generation_cycle):
    
#     out = []
#     context = [0] * block_size # initialize with all ...
#     ix_prev = 0
#     for _ in range(max_new_tokens):
#       # forward pass the neural net
#       logits = model(torch.tensor([context]))
#       probs = F.softmax(logits, dim=1)
#       # sample from the distribution
#       #print(probs,logits)
#       ix = torch.multinomial(probs, num_samples=1).item()
      
#       # shift the context window and track the samples
#       context = context[1:] + [ix]
      
#       if ix_prev!=0 and ix == 0:
#         break
#       ix_prev = ix
#       #if ix < 53:
#       if ix !=0 :    
#           out.append(ix)
#     #out.pop()   
#     #print(''.join(out)) 
#     #print(out)
#     print(sm)
#     print(''.join(itos[i] for i in out))
#     sms = ''.join(itos[i] for i in out)
#     if sms:
#         molecule = Chem.MolFromSmiles(sms)
#         if molecule:
#             valid_smile.append(''.join(itos[i] for i in out))
    
    #print(''.join(itos[i] for i in out)) # decode and print the generated word
 
#%%

sample = valid_smile[0]#'*c1ccc2c(c1)C(=O)N(c1ccc(C(C)(C)c3ccc(N4C(=O)c5ccc(C(*)(C(F)(F)F)C(F)(F)F)cc5C4=O)cc3)cc1)C2=O'# valid_smile[84]

from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole
molecule = Chem.MolFromSmiles(sample)
molecule     
# #%%
# valid_string = ['*CCOC(=O)Sc1ccc(S(=O)(=O)c2ccc(*)cc2)cc1','*CC(*)c1ccc(Cl)cc1','*Oc1ccc(CC(NC(=O)CCc2ccc(OC(=O)CCCCCCCCC(*)=O)cc2)CCCCCCCCCCCCCC)o1','*CCCCCCCCCCOC(=O)c1cccc(C(=O)O*)c1']


#%%

# Save to file
# with open("/Users/rahulkarmakar/Documents/PostDoc/IIT_Madras/Machine_Learning/glass_transition_data/attachments/New_run_after_Random_shuffling/n-gram/generate_smile_vocab_4000.txt", "a") as file:
#     for item in valid_smile:
#         file.write(f"{item}\n")


#%%

print('Valid Smiles %',len(valid_smile)*100/generation_cycle)

#%%

def canonicalize_smiles(smiles_list, batch_size=100):
    """Canonicalize SMILES in batches for large datasets."""
    canonical_smiles = set()
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        for smi in batch:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canonical_smiles.add(Chem.MolToSmiles(mol))
            except:
                print(f"Invalid SMILES: {smi}")
    return canonical_smiles

def compare_datasets(training_smiles, generated_smiles):
    """Compare training and generated SMILES datasets."""
    common_smiles = training_smiles.intersection(generated_smiles)
    unique_generated_smiles = generated_smiles - training_smiles
    return common_smiles, unique_generated_smiles


train_set = text_shuffle[:n1]

# Canonicalize SMILES
training_smiles = canonicalize_smiles(train_set)
generated_smiles = canonicalize_smiles(valid_smile)

# Compare
common_smiles, unique_generated_smiles = compare_datasets(training_smiles, generated_smiles)

# Results
print(f"Number of overlapping SMILES: {len(common_smiles)}")
print(f"Number of unique SMILES: {len(unique_generated_smiles)}")
print(f"Total SMILES: {len(unique_generated_smiles)+len(common_smiles)}")
print(f"Unique SMILES %: {len(unique_generated_smiles)/(len(unique_generated_smiles)+len(common_smiles))}")
#%%%
# Save to file
# with open("/home/tarak/Documents/Rahul/Post-doc/IIT-madras/Machine-Learning/LLM/attachments/Byte_pair_encoding/New_run_after_random_shuffle/Attention/test_smiles.txt", "w") as file:
#     for item in train_set:
#         file.write(f"{item}\n")
