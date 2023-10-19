MeTaCo-XMT
---
### Introduction

---
A Meta-Task Collector-based Cross-lingual Meta-Transfer framework (MeTaCo-XMT) to strategically select support and query set pairs to construct meta training data (meta-tasks). 
Syntactic differences have an effect on transfer performance, so we consider a syntactic similarity sampling strategy and propose a syntactic distance metric model (SDMM) consisting of a syntactic encoder block based on the pre-trained model and a distance metric block using Word Move's Distance (WMD). 

This resource contains two directories ```src``` and ```data```, the SDMM and multilingual task (MRC and NER) models code in ```src```, and all the train and test datasets in ```data```.


### Environment

---
- GPU       NVIDIA GeForce 3090  24G
- python    3.7.13
- torch     1.12.1
- cuda      11.7

### Usage
1. Download the  ```tydiqa``` train data and ```wikiann``` data according to README in their file. 

2.
```
bash run.sh
```