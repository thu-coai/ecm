## Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory

This project is a tensorflow implement of our work, [ECM](https://arxiv.org/abs/1704.01074).

### requirements
 - Python 2.7
 - Tensorflow 0.12
 - Numpy

### Usage
1. **train**
```Shell
python baseline.py --use_emb --use_imemory --use_ememory
```
You can remove "--use_emb", "--use_imemory", "--use_ememory" to remove the embedding, internal memory, and external memory module respectively

2. **test**
```Shell
python baseline.py --use_emb --use_imemory --use_ememory --decode
```
You can test and apply the ecm model using this command. Note: the input word should be splitted by ' ', for example '我 很 喜欢 你 ！', or you can add the chinese text segmentation module in split() function.

### dataset

Due to the copyright of STC dataset, you can ask Lifeng Shang (lifengshang@gmail.com) for the STC dataset ([Neural Responding Machine for Short-Text Conversation](https://arxiv.org/abs/1503.02364v2)), and build the ESTC dataset follow the instruction in the Data Preparation Section of our paper, [ECM](https://arxiv.org/abs/1704.01074).

For your convenience, we also recommand you implement your model using the nlpcc2017 dataset (http://aihuang.org:8000/p/challenge.html), which has more than 1 million Weibo post-response pairs with emotional labels.
