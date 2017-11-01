import json
import sys
import os
from bs4 import BeautifulSoup
import re

if len(sys.argv)<3:
    print("<json corpus> <text corpus>")

json_corpus = sys.argv[1]
text_corpus = sys.argv[2]

#Read in each json file and print each text file for Mallet
for filename in os.listdir(json_corpus):
    with open(os.path.join(json_corpus,filename),'r') as jf:
        data = json.load(jf)
        for doc in data:
            #write file
            filename = filename.split('.')[0]
            docname = doc['id']
            with open(os.path.join(text_corpus,filename,docname),'w') as df:
                df.write(BeautifulSoup(doc['contents'],'html').text)
