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
And then create a `data` directory and download the data there. 

## 2. Train

```
python train.py
```
Also makes a submission file for you. 

## 3. Approach

I've followed ebay's "two tower approach" (https://arxiv.org/pdf/2102.06156.pdf) - encode users as N-dimensional vector, encode articles as N-dimensional vector, train with contrastive loss.

It got awful performance on MAP@12, but manually examining the labels, it seems to be recommending reasonable products, so I'm happy with it. After some grid search, and using a GPU, it should do much better. 

During test time, we need to predict over a million's customers similarities with over 100k articles. I've used Facebook's faiss package (https://www.pinecone.io/learn/faiss-tutorial/). Takes around 0.5 seconds to query 1000 customers, so should be ok for production, if you do some clever batching of requests. 

## 4. Model architecture

I've not used the images; nor the textual descriptions of the items. I just selected some categorical features for articles and for customers, and did 10-dimensional embeddings on them. Then concatenated those embeddings, and put a few fully-connected layers on top. 

The loss function is cosine similarity, as in Ebay's paper; faiss supports dot product distance, so that's fine. 

## 5. Further work

* Train on GPU. training on CPU is pretty slow. 
* Add dropout after embedding layers. I've seen a few implementations do that.
* Take advantage of images + text descriptions of items. 
* Do grid search on the depth and width of the two towers. 
* Play around with the possible learning rates for the two towers. Since we have much fewer items per customer, that customers per item, we might like a higher learning rate.

## 6. Other possible approaches:

* Use Facebook's `torchrec` package. It can scale excellently, it's the same two-tower approach; but it's in beta, and relies on a whole bunch of other packages. So it's definitely a time investment.
* Use graph networks like in this Uber blogpost: https://www.uber.com/en-BG/blog/uber-eats-graph-learning/. 
* Use the classical collaborative filtering. Should be a solid, simple baseline. 