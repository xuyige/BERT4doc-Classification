import pandas as pd
import spacy
import numpy as np

nlp=spacy.load("en_core_web_md")

test_data=pd.read_csv("test.csv",header=None,sep=",").values
train_data=pd.read_csv("train.csv",header=None,sep=",").values

test=[]
train=[]
with open("AGnews_corpus_test.txt","w",encoding="utf-8") as f_test:
    with open("AGnews_corpus_train.txt", "w", encoding="utf-8") as f_train:
        with open("AGnews_corpus.txt","w",encoding="utf-8") as f:
            for i in range(len(test_data)):
                if i%1000==0:print(i)
                f.write(str(test_data[i][1])+"\n")
                f_test.write(str(test_data[i][1])+"\n")
                document=nlp(str(test_data[i][2]))
                number=0
                for sent in document.sents:
                    number+=1
                    f.write(str(sent)+"\n")
                    f_test.write(str(sent)+"\n")
                test.append(number)
                f.write("\n")
                f_test.write("\n")

            for i in range(len(train_data)):
                if i%1000==0:print(i)
                f.write(str(train_data[i][1])+"\n")
                f_train.write(str(train_data[i][1])+"\n")
                document=nlp(str(train_data[i][2]))
                number=0
                for sent in document.sents:
                    number+=1
                    f.write(str(sent)+"\n")
                    f_train.write(str(sent) + "\n")
                train.append(number)
                f.write("\n")
                f_train.write("\n")


print("test_max=",np.max(test))
print("test_avg=",np.average(test))
print()
print("train_max=",np.max(train))
print("train_avg=",np.average(train))


