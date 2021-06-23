from ckiptagger import WS, POS, NER
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm, trange


os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# Training & testing data load
excel_data_dir = "data"
train_data_path = "train.xlsx"
test_data_path = "test.xlsx"

df_train = pd.read_excel(os.path.join(excel_data_dir, train_data_path), engine='openpyxl')
df_test = pd.read_excel(os.path.join(excel_data_dir, test_data_path), engine='openpyxl')

contents = df_test["content"]

print("Load model")
# Load ckip model
ws = WS("./data") # Should download CkipTagger's  
pos = POS("./data")
ner = NER("./data")


# CKIP functions
def find_people(entity_sentence_list):
    people_list = []

    for entity in sorted(entity_sentence_list):
        if entity[2] == "PERSON":
            people_list.append(entity[3])
    return people_list


def find_fraud_works(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    fraud_key_word = ['詐欺', '詐欺犯', '吸金', '不法獲取', '不法獲利', '高利貸', '不法所得',
                  '抵押', '炒股', '炒作', '內線交易' , '營業秘密', '著作權法', '商業機密',
                  '貪汙治罪', '背信罪', '詐騙', '捲款', '捲走', '捲款潛逃', '證交法',
                  '挪用', '盜用', '貪汙', '收賄', '詐貸', '畫大餅', '誘騙', '騙取', '哄騙']
    fraud_word_list = []
    for word, pos in zip(word_sentence, pos_sentence):
        if pos == 'VC':
            for fkw in fraud_key_word:
                if word == fkw:  
                    fraud_word_list.append(word)
    return fraud_word_list



def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        #if pos == 'VC':
        print(f"{word}({pos})", end="\u3000")
    print()
    return

tmp_contents = contents

print("Ready to processing!!!")
pbar = tqdm(total=len(tmp_contents))
# Processing: Detect people and frawd words
contents_people = [] # list with lists
contents_fraud = [] # list with lists
for content in tmp_contents:
    word_sentence_list = ws([content])
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    
    people_list = []
    fraud_word_list = []
    for i, sentence in enumerate([content]):
        fraud_word = find_fraud_works(word_sentence_list[i], pos_sentence_list[i])
        if len(fraud_word) != 0:
            for word in fraud_word:
                fraud_word_list.append(word)
            people = find_people(entity_sentence_list[i])
            for person in people:
                people_list.append(person)

    contents_people.append(people_list)
    contents_fraud.append(fraud_word_list)
    pbar.update(1)

assert len(contents_people) == len(tmp_contents)

#print(contents_people)

print("Dump to json file!!!!")

# Dump to json file
with open('output/test/people.json', 'w', newline="", encoding='utf8') as jsonfile:
    json.dump(contents_people, jsonfile)
with open('output/test/fraud_word.json', 'w', newline="", encoding='utf8') as jsonfile:
    json.dump(contents_fraud, jsonfile)

# Generating predition
with open('result/test/people.json') as jsonfile:
    data = json.load(jsonfile

prediction_test = np.array(data, dtype=object)
pd.DataFrame(prediction_test).to_excel("prediction.xlsx", header=["label"], index=None)








