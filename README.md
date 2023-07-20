# Comparable-partallel-Algo2
Fuzzy Influenced Process to Generate Comparable to Parallel Corpora


This algorithm aligns sentences of comparable corpora using expectation maximization mentioned in Richa Kulkarni Thesis. 
Then filters sentences using ibm model 1 score mentioned in Mining Large-scale Parallel Corpora from Multilingual Patents: An English-Chinese example and its application to SMT by Bin Lu, Benjamin K. Tsou, Tao Jian, Oi Yee Kwong, and Jingbo Zhu

To install all dependencies execute dependencies.sh

First run this to use hindi stemmer:

cd morphanlyser/
./run.sh 

execute in new terminal
python3 em_and_filter.py eng_corpus hin_corpus

It outputs in output dirctory
Make sure pickle folder in same directory



# Cite:
Fuzzy Influenced Process to Generate Comparable to Parallel Corpora
DEBAJYOTY BANIK, ASIF EKBAL, SURESH CHANDRA SATAPATHY, ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 2023
