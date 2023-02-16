# HandM_recommend
Simple recommendation engine for https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations


## 1. Set up environment

```
# build docker image
cd docker
. build.sh

# run container
docker run -it -d --name handm --net=host --ipc=host --gpus all -v ~/HandM_recommend:/HandM_recommend handm:1.0
docker exec -it handm bash
```

## 2. Approach

I've followed ebay's "two tower approach" (https://arxiv.org/pdf/2102.06156.pdf) - encode users as N-dimensional vector, encode articles as N-dimensional vector, train with contrastive loss.

It got awful performance on MAP@12, but manually examining the labels, it seems to be recommending reasonable products, so I'm happy with it. After some grid search, and using a GPU, it should do much better. 

During test time, we need to predict over a million's customers similarities with over 100k articles. I've used Facebook's faiss package (https://www.pinecone.io/learn/faiss-tutorial/). Takes around 0.5 seconds to query 1000 customers, so should be ok for production, if you do some clever batching of requests. 

## 3. Model architecture

I've not used the images; nor the textual descriptions of the items. I just selected some categorical features for articles and for customers, and did 10-dimensional embeddings on them. Then concatenated those embeddings, and put a few fully-connected layers on top. 

The loss function is cosine similarity, as in Ebay's paper; faiss supports dot product distance, so that's fine. 

## 4. Further work

4.1 Train on GPU. training on CPU is pretty slow. 
4.2 Add dropout after embedding layers. I've seen a few implementations do that.
4.3 Take advantage of images + text descriptions of items. 
4.4 Do grid search on the depth and width of the two towers. 
4.5 Play around with the possible learning rates for the two towers. Since we have much fewer items per customer, that customers per item, we might like a higher learning rate.