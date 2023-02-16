import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from typing import Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import randint

from tqdm import tqdm
from einops import rearrange
import faiss
from metrics import apk


def reduce_unique_values(dataframe: pd.DataFrame, col_max_counts: Dict) -> Dict:
    '''
    Takes as input a dataframe, with some columns we'd like embeddings of,
    and maximum unique values per column.
    
    Changes the dataframe inplace, so that only the top n values for each column remain,
    and returns the mappings string -> idx for each column. 
    '''
    
    mappings = {}
    for col in col_max_counts:
        top_n = dataframe[col].value_counts().head(col_max_counts[col]-1).index.to_list() # minus one, to leave room for "other"
        # change all other values to "other"
        if dataframe[col].nunique() > col_max_counts[col]:
            dataframe.loc[~dataframe[col].isin(top_n), col] = 'other'
            top_n.append('other')
        mappings[col] = {top_n_attr: idx for (top_n_attr, idx) in zip(top_n, range(len(top_n)))}
    return mappings


transactions = pd.read_csv('data/transactions_train.csv',
                           dtype={'article_id': str})

articles = pd.read_csv('data/articles.csv', 
                       dtype={'article_id': str})
articles.set_index('article_id', inplace=True, drop=False)
select_top_n = {
    'prod_name': 500,
    'product_type_name': 500,
    'product_group_name': 100,    
    'graphical_appearance_name': 100,
    'colour_group_name': 100, 
    'perceived_colour_value_name': 100,
    'perceived_colour_master_name': 100,
    'department_name': 230,
    'index_name': 100,
    'garment_group_name': 100,
    'section_name': 100,
    'garment_group_name': 100
}
mappings_articles = reduce_unique_values(articles, select_top_n)
mappings_articles['perceived_colour_value_name']


customers = pd.read_csv('data/customers.csv')
customers.age.fillna(32, inplace=True) # median age
customers.set_index('customer_id', inplace=True, drop=False)
select_top_n = {
    'customer_id': 100_000_000, # separate embedding for each customer. 
    'age': 100, # all ages are embedded, unless we have really old people. 
    'postal_code': 50_000 # 5 or 6 users for the smallest codes remaining. 
}
mappings_customers = reduce_unique_values(customers, select_top_n)

class GroupEmbedding(nn.Module):
    '''
    Given something with a lot of thing we want embeddings of - 
    for example, articles of clothing, or customers,
    and mappings string->index, creates torch.nn.Embedding layers,
    and concatenates their outputs
    '''
    
    def __init__(self, mappings: Dict):
        super().__init__()
        self.mappings = mappings
        self.embeddings = {}
        for col in mappings:
            embedding_dim = 10
            self.embeddings[col] = nn.Embedding(num_embeddings = len(mappings[col]), embedding_dim=10)
            for param in self.embeddings[col].parameters():
                self.register_parameter(name=f'embedding_{col}', 
                                        param=param)
        
    def forward(self, dataframe: pd.DataFrame) -> torch.Tensor:
        embedded_columns = []
        for col in self.mappings:
            strings = dataframe[col]
            indexes = torch.tensor([self.mappings[col][item] for item in strings], dtype=torch.long)
            embedded_columns.append(self.embeddings[col](indexes))
            
        return torch.cat(embedded_columns, axis=1)
 
article_embedder = GroupEmbedding(mappings_articles)
customer_embedder = GroupEmbedding(mappings_customers)

class TransactionDataLoader():
    def __init__(self, 
                 customers: pd.DataFrame, 
                 articles: pd.DataFrame, 
                 txns: pd.DataFrame, 
                 negative_samples: int = 10):
        self.customers = customers
        self.articles = articles
        self.txns = txns
        self.negative_samples = negative_samples # per one positive sample
    
    def __len__(self):
        return len(self.txns)
    
    def get_batch(self, batch_size: int = 1000):
        batch_index_start = randint(0, len(self) - batch_size)
        # for speed purposes, do not sample randomly, but give data iteratively
        batch_txns = self.txns.iloc[batch_index_start:batch_index_start + batch_size]
        batch_customers = self.customers.loc[batch_txns.customer_id]
        positive_articles = self.articles.loc[batch_txns.article_id]

        # now, negative sampling:
        negative_articles = self.articles.sample(batch_size*self.negative_samples)
        labels = torch.ones(batch_size*(1+self.negative_samples))
        labels[batch_size:] = -1
        batch_customers = pd.concat([batch_customers for _ in range(self.negative_samples + 1)])
        batch_articles = pd.concat([positive_articles, negative_articles])
        return batch_customers, batch_articles, labels
    
tdl = TransactionDataLoader(customers, articles, transactions, negative_samples=5)
customer_tower = nn.Sequential(
    customer_embedder,
    nn.ReLU(),
    nn.Linear(30, 128),
    nn.Linear(128, 128)
)

article_tower = nn.Sequential(
    article_embedder,
    nn.ReLU(),
    nn.Linear(110, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128)
)

# we want a higher learning rate, since we have a lot fewer datapoints
optimizer_customers = torch.optim.Adam(params=customer_tower.parameters(), 
                                       lr = 0.05) 
optimizer_articles = torch.optim.Adam(params = article_tower.parameters(),
                                      lr = 0.001)

writer = SummaryWriter(log_dir='logs/full_model')

cosine_loss = nn.CosineEmbeddingLoss()
for batch in tqdm(range(2_000_000)): 
    batch_cust, batch_articles, batch_labels = tdl.get_batch()
    cust_emb = customer_tower(batch_cust)
    art_emb = article_tower(batch_articles)
    loss = cosine_loss(
        art_emb,
        cust_emb,
        batch_labels
    )
    loss.backward()
    optimizer_articles.step()
    optimizer_customers.step()
    optimizer_articles.zero_grad()
    optimizer_customers.zero_grad()
    writer.add_scalar("Loss/train", loss.item(), batch)
    
all_article_embeddings = article_tower(articles)
## normalize by norm
all_article_embeddings = torch.div(all_article_embeddings, 
                                   rearrange(torch.linalg.norm(all_article_embeddings, dim=1), 'w -> w 1'))
index = faiss.IndexFlatIP(128)
index.add(all_article_embeddings.detach().numpy())
# faiss uses the order in which we put them in the index. 
# So let's note that down. 
articles['idx'] = range(len(articles)) 

# loop over the submission file and put in our predictions
submission = pd.read_csv('data/sample_submission.csv')
articles = articles.set_index('idx')
batch_size_faiss = 1000
current_customer = 0
while current_customer < len(submission):
    current_customer += batch_size_faiss
    batch_submission = submission.iloc[current_customer:current_customer+batch_size_faiss]
    batch_customers = customers.loc[batch_submission.customer_id]
    batch_customers = customer_tower(batch_customers)
    batch_customers = torch.div(batch_customers, 
                                rearrange(torch.linalg.norm(batch_customers, dim=1), 'w -> w 1'))
    predicted_products = index.search(batch_customers.detach().numpy(), k=12)[1]

    ## convert from idx to article_id:
    predicted_product_ids = []
    for customer_predicted in predicted_products:
        predicted_product_ids.append(' '.join(articles.loc[customer_predicted].article_id.astype('str')))

    submission.loc[batch_submission.index, 'prediction'] = predicted_product_ids
    
submission.to_csv('predicted_submission.csv', index=False)