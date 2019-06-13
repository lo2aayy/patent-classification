# patent-classification
Implementation of "Optimizing neural networks for patent classification" paper for wipo-alpha dataset

## How to install

Install the following requirements:
  - python3
  - pyfasttext
  - keras
  
Download Wipo-alpha dataset and put extracted folder in resources
Download fasttext word embedding and put in resources

Create the dataset by executing:

```
python create_dataset -train
python create_dataset -test
```

Train model by executing:
```
python train.py -embedding <embedding_name>
```
