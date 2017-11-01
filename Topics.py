# -*- coding: utf-8 -*- 
print "Importing Libraries..."
import re
import glob
import copy
import operator

import math
import json
import csv

import os

#from scipy.stats import entropy
#from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import codecs
import gensim
import sys
import time
from datetime import datetime
from collections import Counter

import urllib2,urllib

from collections import defaultdict


os.system('cls')

startTime = datetime.now()
lastTime = startTime
print "\n \n", "Pre-processing and LDA topic modeling,,,"
global last_class
global topicWordTags

main_events = ["Doc_open", "Highlight", "Search", "CreateNote", "Connection"]

__saved_context__ = {}

def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]


def is_int(s):
    try:     
        ss = str(s)
        #if len(ss) == len(ss.strip({0,1,2,3,4,5,6,7,8,9})):
        if re.search('\d+', ss):
            return True
        else:			
            return False			
    except ValueError:			
        return False	


def LDA_Topic(Int_type, de_stemmer, corp,Text_lda1, my_dictionary,Text_tfidf):  
    # Defines LDA topic number for search terms/notes/highlights/etc/etc.
    # ------------------- 1 Stop words----------------------
    
    # <span class="highlight-pink">Cato</span>']
    
    # print "Input: ", corp
    #raw = re.sub("\d+","",raw)
    #raw = raw.replace("’","'")    
    English_stop_words = get_stop_words('en')
    My_list = ['span', 'highlight', 'pink','class', 'one','two','three','four','five','six','seven','eight','nine','ten', '://' ,'http', 'www' ,'com', 'don', 'pre', 'paid', 'must', 'tcan',  'twhen', 'twhat', 'via','are', 'will' ,'said', 'can', 'near', 'and', 'the', 'i', 'a', 'to', 'it', 'was', 'he', 'of', 'in', 'you', 'that', 'but', 'so', 'on', 'up', 'we', 'all', 'for', 'out', 'me', 'him', 'they', 'says', 'got', 'then', 'there', 'no', 'his', 'as', 'with', 'them', 'she', 'said', 'down', 'see', 'had', 'when', 'about', 'what', 'my', 'well', 'if', 'at', 'come', 'would', 'by', 'one', 'do', 'be', 'her', "didn't", 'jim', 'get', "don't", 'time', 'or', 'right', 'could', 'is', 'went', "warn't", "ain't", 'good', 'off', 'over', 'go', 'just', 'way', 'like', 'old', 'around', 'know', 'de', 'now', 'this', 'along', 'en', 'done', 'because', 'back', "it's", 'tom', "couldn't", 'ever', 'why', 'going', 'little', 'some', 'your', 'man', 'never', 'too', 'more', 'say', 'says', 'again', 'how', 'here', 'tell', 'posted' , 'need' , 'needs' , 'someone', 'government', 'intelligence', 'report']
    
    stoplist_1 = set('a b c d e f g h i j k l m n o p q r s t u v w x y z 1 2 3 4 5 6 7 8 9 0'.split(' ')) # Create a set of enlighs alphabets
    stoplist_2 = set(); #English_stop_words)
    stoplist_3 = set('es la . , . <br> <br><br> br > : >< < .< { } [ ] ( ) .' '\' ` " “ ” ? ! - \u201d< \u201d .\u201d \u201d u201d \u2019 \xe9 !< >!'.split(' ')) # Create a set 
    stoplist_4 = set(My_list)
    
    stoplist = 	stoplist_1 | stoplist_2 | stoplist_3 | stoplist_4
    # ------------------- 2 tokenizer ----------------------
    stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') &(word != u'.\u201d') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
        for document in corp]
    # stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') &(word != u'.\u201d') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
        # for document in corp]
		
	# ------------------- 3 Stemming and Count word frequencies -------------------
    p_stemmer = PorterStemmer()
    stemmer = {}              
    texts = []	
    texts_set = []
    
    for stopped_token in stopped_tokens:
        stemmed_texts = [p_stemmer.stem(i) for i in stopped_token]
        texts_set += [stemmed_texts]		
		
    frequency = defaultdict(int)
    for text in texts_set:
        for token in text:
            frequency[token] += 1
			
    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts_set]

    # ------------------- 4 Dictionary and TF-IDF Vectors -------------------    
    ids2words = my_dictionary.token2id
    bow_corpus = [my_dictionary.doc2bow(text) for text in processed_corpus]

    # print "------------------>>>>: ", corp
    all_vectors = Text_tfidf[bow_corpus] #bow_corpus]   # Gives representative vectors 	  

    # ------------------- 5 Document Vectors and Classification -------------------    

    counter = []
    doc_topics = []

    for each in range(0, class_num):
        counter.append(0)

    for index, document in enumerate(all_vectors):    # Each documents probability to calss
        doc_topics.append(Text_lda1.get_document_topics(document)) # , minimum_probability=0.19)
        new_list = []		
        for each_topic in doc_topics[-1]:
            new_list.append(each_topic[1])
        t_index, value = max(enumerate(new_list), key=operator.itemgetter(1))		
        
    # ------------------- 6 Word tags -------------------    
    new_list = []	
    key_words = []
    i = 0 
    # Words from doc TF-IDF Vector
    # Sort word bag of each document 
    if len(all_vectors[0]) > 3: 
        new_list = sorted(all_vectors[0], key=lambda prob: prob[1], reverse=True)
    else:
        new_list = all_vectors[0]
    # print "Topic: ", t_index + 1
        
    for i in range(0,len(new_list)):   # Pick the firts 5 keywords in sorted list
        for key in ids2words:
            if ids2words[key] == new_list[i][0]: # bow_corpus[1][2][0]:
                if (i<3):              # first 3 keywords, no more 
                    term = de_stemmer[key]                    				
                    key_words.append(str(term))
                    topicWordTags[t_index + 1].add(str(term))   # Add this to the bag of words  

    # print "summary: ", key_words
    
    # ------------------- 6 Final Word tags and sorting -------------------
    # temp = [""]
    temp = corp[0]
    if Int_type == "Search":  # 
        finalBag[t_index + 1] = finalBag[t_index + 1] + ' ' + temp[0] + ' ' + temp[0] + ' ' + temp[0]
        # print "Search: ", temp #finalBag[t_index + 1]
    elif Int_type == "Add note":
        finalBag[t_index + 1] = finalBag[t_index + 1] + ' ' + temp[0] + ' ' + temp[0]
    else:
        finalBag[t_index + 1] = finalBag[t_index + 1] + ' ' + temp[0]
    
    # finalBag[t_index + 1] = finalBag[t_index + 1] + ' ' + temp[0]   #bow_corpus   # Keep adding user interactions string entities...!
    # print "Bag: ", finalBag
    
    # if EntList == 1:          # TF-IDF and sort entities list for each topic
    
        # for j in range(0,len(finalBag)): 
            # final_vectors = Text_tfidf[finalBag[j]] #bow_corpus]   # Gives representative vectors 	  
            # print "\n Bag of words: ", finalBag[j]
            # print "topic#: ", j
            # new_list = []	
            # key_words = []
            # # Words from doc TF-IDF Vector
            # # Sort word bag of each document 
            # if len(final_vectors[0]) > 2: 
                # new_list = sorted(final_vectors[0], key=lambda prob: prob[1], reverse=True)
            # else:
                # new_list = all_vectors[0]
            # print "\n Bag of words/sorted: ", new_list

            # for i in range(0,len(new_list)):   # Pick the firts 10 keywords in sorted list
                # for key in ids2words:
                    # if ids2words[key] == new_list[i][0]: # bow_corpus[1][2][0]:
                        # if (i<10):              # first 3 keywords, no more 
                            # term = de_stemmer[key]                    				
                            # topicWordTags2[j].add(str(term))   # Add this to the bag of words  
            
            # print "\n Final Entites: ", topicWordTags2

    return t_index + 1
	
def LDA_Topic_Clustering(corp,reading_weight, new_model,class_num,LDA_passes,x,y):

    # ------------------- 1 Stop words----------------------
    #raw = re.sub("\d+","",raw)
    #raw = raw.replace("’","'")    
    English_stop_words = get_stop_words('en')
    My_list = [".'",".']","]']","\'\'", 'one','two','three','four','five','six','seven','eight','nine','ten', '://' ,'http', 'www' ,'com', 'don', 'pre', 'paid', 'must', 'tcan',  'twhen', 'twhat', 'via','are', 'will' ,'said', 'can', 'near', 'and', 'the', 'i', 'a', 'to', 'it', 'was', 'he', 'of', 'in', 'you', 'that', 'but', 'so', 'on', 'up', 'we', 'all', 'for', 'out', 'me', 'him', 'they', 'says', 'got', 'then', 'there', 'no', 'his', 'as', 'with', 'them', 'she', 'said', 'down', 'see', 'had', 'when', 'about', 'what', 'my', 'well', 'if', 'at', 'come', 'would', 'by', 'one', 'do', 'be', 'her', "didn't", 'jim', 'get', "don't", 'time', 'or', 'right', 'could', 'is', 'went', "warn't", "ain't", 'good', 'off', 'over', 'go', 'just', 'way', 'like', 'old', 'around', 'know', 'de', 'now', 'this', 'along', 'en', 'done', 'because', 'back', "it's", 'tom', "couldn't", 'ever', 'why', 'going', 'little', 'some', 'your', 'man', 'never', 'too', 'more', 'say', 'says', 'again', 'how', 'here', 'tell', 'message', 'posted' , 'need' , 'needs' , 'someone', 'government', 'intelligence', 'report']
    
    stoplist_1 = set('a b c d e f g h i j k l m n o p q r s t u v w x y z 1 2 3 4 5 6 7 8 9 0'.split(' ')) # Create a set of enlighs alphabets
    stoplist_2 = set(English_stop_words)	
    stoplist_3 = set('es la . , . <br> <br><br> br > : >< < .< { } [ ] ( ) ,\'\'  ." ` " ? ! - \u201d< \u201d .\u201d \u201d u201d \u2019 \xe9 !< >!'.split(' ')) # Create a set 
    #stoplist_33 = set(' .' .'] '.split(' ')) # Create a set 
    stoplist_4 = set(My_list)
    
    stoplist = 	stoplist_1 | stoplist_2 | stoplist_3 | stoplist_4
    # ------------------- 2 tokenizer ----------------------

    stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') &(word != u'.\u201d') & (word != u'\u201c') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
        for document in corp]
	
    # ------------------- 3 Stemming and Count word frequencies -------------------
    p_stemmer = PorterStemmer()
    stemmer = {}              
    texts = []	
    texts_set = [] #set()
    de_stemmer = {}
  
    for stopped_token in stopped_tokens:
        stemmed_texts = [p_stemmer.stem(i) for i in stopped_token]
        texts_set += [stemmed_texts]		
    #texts_set = stopped_tokens    # Without stemmer  
    
    for j in range(0,len(texts_set)):
        for i in range(0,len(texts_set[j])):
            if not texts_set[j][i] in de_stemmer:
                de_stemmer[texts_set[j][i]] = stopped_tokens[j][i]    # Save it later for de_stemmer!
		
    frequency = defaultdict(int)
    for text in texts_set:
        for token in text:
            frequency[token] += 1
			
    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts_set]
    #print processed_corpus 
    #return 0
    # ------------------- 4 Dictionary and TF-IDF Vectors -------------------    
    my_dictionary = corpora.Dictionary(processed_corpus)
    ids2words = my_dictionary.token2id
    bow_corpus = [my_dictionary.doc2bow(text) for text in processed_corpus]

    # ------------------- Add user interactions weights ----------------------    
    i = 0
    new_corp = []
    for each_doc in bow_corpus:
        new = []	
        for each_word in each_doc:
            new.append((each_word[0], each_word[1]*(1+reading_weight[i]))) # new.append(each_word[0], float(each_word[1]) * (1+reading_weight[i]))
            j+=1			
        new_corp.append(new)			
        i+=1
    # train the model
    Text_tfidf = models.TfidfModel(new_corp)
    
    # all_vectors = Text_tfidf[new_corp] #bow_corpus]   # Gives representative vectors 	  
    
    # print "\n bow_corpus: ", bow_corpus
    # print "\n new weighted Docs: ", new_corp
    # print "\n TF-IDF vectors: ", all_vectors
    # print "\n TF-IDF size: ", len(all_vectors)
    # print "\n TF-IDF zero: ", all_vectors[0]
    # for doc in all_vectors:
        # print "each: ", doc
    
    # print "\n The End "
    
    # ------------------- 5 LDA Model and -------------------    
    
    if os.path.isfile("./LDAmodels/LDAmodel_dataset" + str(x) + "_P" + str(y) + "_class" +str(class_num)+".lda") ==0 or (new_model == 1):  # Do you want to train the model?
        print "\n LDA Model Training..."   
        Text_lda = models.LdaModel(new_corp, id2word=my_dictionary, num_topics= class_num, passes = LDA_passes)		#  with out TF-IDF model
        Text_lda.save("./LDAmodels/LDAmodel_dataset" + str(x) + "_P" + str(y) + "_class" + str(class_num) + ".lda") # same for tfidf, lsa, ...
    else:		
        print "\n LDA Model Loading..."
        Text_lda = models.LdaModel.load("./LDAmodels/LDAmodel_dataset" + str(x) + "_P" + str(y) + "_class" +str(class_num)+".lda")			
        
    # ------------------- 6 Document Vectors and Classification -------------------    

    counter = []
    doc_topics = []
	
    for each in range(0, class_num):
        counter.append(0)
		
    for index, document in enumerate(all_vectors):    # Each documents probability to calss
        # infer topic distribution for each document
        doc_topics.append(Text_lda.get_document_topics(document)) # , minimum_probability=0.19)
    #        No_Topic = 1
        new_list = []		
        for each_topic in doc_topics[-1]:
            new_list.append(each_topic[1])
            
        t_index, value = max(enumerate(new_list), key=operator.itemgetter(1))		
        
        # print "\n index: ", t_index
        # print "\n doc_topics: ", doc_topics
        counter[t_index] += 1		
    # ------------------- 7 Create a bag for topic keywords -------------------            
    topicWordTags = []
    topicWordTags2 = []
    topicWordTags3 = []
    
    finalBag = []
    for each in range(0, class_num + 1):
        topicWordTags.append(set())
        topicWordTags2.append([])
        topicWordTags3.append([])  #set())
        finalBag.append('')
    # print "\n Topic word empty: ", topicWordTags    

    # ------------------- 8 Topic summary output -------------------            
    output_topics = Text_lda.show_topics(num_topics=class_num, num_words=15, formatted=False)   # To review topics and terms individually 
    
    return finalBag, topicWordTags, topicWordTags2, topicWordTags3, de_stemmer, ids2words, all_vectors, Text_lda, my_dictionary, Text_tfidf, output_topics, de_stemmer, doc_topics

def save_topic_docs(EntList, my_dictionary1, docs_number, ids2words, doc_vectors, output_topics, doc_topics, de_stemmer, class_num, keyword_num, filename, filename2, filename3):   # Print and save document topics/topic defenitions 

    json_hash = []
    doc_topic = []
    doc_index = 0
    total_prob = []
    interference = []
    doc_topic_array	= []
    doc_key_word = []	
    doc_topic_keywords = []
	
    for each in range(0, class_num):
        doc_topic.append(0)
        total_prob.append(0)
        doc_topic_keywords.append("");
    doc_topic_keywords.append("");    # one more time 
        
    for each in range(0, docs_number):		
        interference.append(0)
    doc_no=0	
    # ------------------------------ Create Document Topics# and Summary ------------------------------
    for each_doc in doc_topics:

        new_list = []	
        key_words = []
		# Words from doc TF-IDF Vector
        # Sort word bag of each document 
        new_list = sorted(doc_vectors[doc_index] , key=lambda prob: prob[1], reverse=True)

        for i in range(0,keyword_num):   # Pick the firts 5 keywords in sorted list
            # print "new_list: ", new_list[i][0]
            for key in ids2words:
                if ids2words[key] == new_list[i][0]	: # bow_corpus[1][2][0]:
                    #print new_list[i][0]
                    term = de_stemmer[key]                    				
                    key_words.append(str(term))

        new_list = []		
        for each_topic in each_doc:
            new_list.append(each_topic[1])
                			
        topic_index, value = max(enumerate(new_list), key=operator.itemgetter(1))  #  topic_index + 1 
        total_prob[topic_index] += value
        
        doc_topic_keywords[topic_index + 1] = doc_topic_keywords[topic_index + 1] + " " + key_words[0] +  " " + key_words[1] +  " " + key_words[2] +  " " + key_words[3] +  " " + key_words[4]
        
        j=0			
        interference[doc_no] = []		
        #        interference[doc_no].append(topic_index) 			
        for each in new_list:
            if (each > 0.1):
                interference[doc_no].append(j) 			
            j+=1				
			
        temp = {"docName": name+" "+str(doc_index+1), "classNum": topic_index+1, "events": key_words}
        json_hash.append(temp)	
		
        doc_topic[topic_index] += 1		
        doc_no += 1
        #>>> print "\n","Doc #",str(doc_index+1),", Topic: ",topic_index,"Prob.: ", value, " Vector: ", each_doc, key_words
        doc_topic_array.append(topic_index + 1)		# <<<<<<<<<<< To here 
        doc_key_word.append(key_words)     # save key words

        doc_index += 1			
    #>>>print "inter: ",  interference        		
    
    #>>>>print "\n", "Doc/Topics list: ", doc_topic, 'Total: ',sum(list(doc_topic))
    
    #>>>>print "\n", "Total Doc/Topics probability: ",[x/y for x, y in zip(total_prob, doc_topic)], ", Overal: ", sum(list(total_prob))	/sum(list(doc_topic))

    #>>>>for id,terms in output_topics:
    #>>>>print "\n","Topic ",str(id),"/ Docs: ",doc_topic[id] , ":", terms

    fout = open(filename,"w")
    fout.write(json.dumps(json_hash,indent=1))
    fout.close()

    # ------------------------------ Create LDA Topic Summary ------------------------------
    if EntList == 1: 
        compare_hash = []
        
        for i in range(0,class_num): 
            answer = []
            bb = []
            for each in output_topics[i]:
                # print "\n each: ", each
                if each < 10:
                    aa = each
                else:
                    for eachh in each:
                        bb.append(eachh[0])
                
            temp = {"TopicNum: ": aa, "keywords": bb}
            # print "\n temp: ", temp
            compare_hash.append(temp)
        
        fout = open(filename2,"w")
        fout.write(json.dumps(compare_hash,indent=1))
        fout.close()
    
    # ----------------------------------- User Entities List ----------------------------------------

    
    if EntList == 1:          # TF-IDF and sort entities list for each topic
    
        English_stop_words = get_stop_words('en')
        My_list = ["u'\u201c'", 'span', 'highlight', 'pink','class', 'one','two','three','four','five','six','seven','eight','nine','ten', '://' ,'http', 'www' ,'com', 'don', 'pre', 'paid', 'must', 'tcan',  'twhen', 'twhat', 'via','are', 'will' ,'said', 'can', 'near', 'and', 'the', 'i', 'a', 'to', 'it', 'was', 'he', 'of', 'in', 'you', 'that', 'but', 'so', 'on', 'up', 'we', 'all', 'for', 'out', 'me', 'him', 'they', 'says', 'got', 'then', 'there', 'no', 'his', 'as', 'with', 'them', 'she', 'said', 'down', 'see', 'had', 'when', 'about', 'what', 'my', 'well', 'if', 'at', 'come', 'would', 'by', 'one', 'do', 'be', 'her', "didn't", 'jim', 'get', "don't", 'time', 'or', 'right', 'could', 'is', 'went', "warn't", "ain't", 'good', 'off', 'over', 'go', 'just', 'way', 'like', 'old', 'around', 'know', 'de', 'now', 'this', 'along', 'en', 'done', 'because', 'back', "it's", 'tom', "couldn't", 'ever', 'why', 'going', 'little', 'some', 'your', 'man', 'never', 'too', 'more', 'say', 'says', 'again', 'how', 'here', 'tell', 'posted' , 'need' , 'needs' , 'someone', 'government', 'intelligence', 'report']
        
        stoplist_1 = set('a b c d e guy f size styled g h also number details since due countries using selling sent given earlier completely owed full player numerous thus recovered number i j k unknown move l m n o p q r else s t u v w x y z first becomes able actually absolutely necessary officialise entire stage issued'.split(' ')) # Create a set of enlighs alphabets
        stoplist_2 = set(English_stop_words)
        stoplist_3 = set('es la . , . taken <br> however require ratio note illumination homeland give order possibly think questions event hour case occurred yet confirmed destination million want update arrived removed responsibility known claiming icon role display none stating closed work apply research provided additional closed caused showed month succeeded knowledge stop coroner style index enclosed sudden seeks wait last soon centers outside believed feet happened begins colors hour people airing large claims area getting blkd highly whose young information made year ptf create make public date text tried space found name run ome ngoki agree everyone caller identification <br><br> br > : >< < .< { } [ ] ( ) .' '\' ` " “ ” ? ! - \u2018 \xe9 \u201c \u201d< \u201d .\u201d \u201d u201d \u201c looking .\u201d< \u2019 worth realized facilitated \xe9 keeping !< >! ago note sending'.split(' ')) # Create a set 
        stoplist_4 = set(My_list)
         
        stoplist = 	stoplist_1 | stoplist_2 | stoplist_3 | stoplist_4
        
        p_stemmer = PorterStemmer()
        # ----------------- Process wordtags from user interactions -----------------------------------
        timeList = [ 'date', 'jan', 'january', 'feb', 'february' , 'march', 'april' , 'may' , 'present', 'jun', 'july', 'august', 'september', 'october', 'november', 'december', '1998']
        placeList = ['engstrom','gastech','abila','kronos','petra','jet','limousine','tethan','','','','headquarters','tethys','elodis','airport','airports', 'vastopolis', 'terrorist', 'brotherhood', 'antarctica', 'washington', 'dhs','valujet','laboratory','dharan','bahrain','qatar','kuwait','airlines','vastpress','ibm','suburbia','bruno','lab','antarctica','nigeria', 'dubai', 'burj', 'syria', 'gaza', 'sanaa', 'ebilaead' , 'tabriz' , 'venezuela' , 'pakistan' , 'countries' , 'saudi' , 'arabia' , 'kenya', 'iran' , 'lebanon' , 'russia', 'yemen' , 'turkey', 'arkadi', 'barcelona', 'paris', 'cafe', 'mosque' , 'exhibition', 'valley', 'moscow', 'downtown', 'mombasa', 'bangkok', 'sudan' , 'usa', 'washington' , 'milan', 'italy' , 'hospital', 'british' , 'soviet' , 'antalya', 'malaysia' , 'somalia','sana' ,'lagos','pyongyang','uae', 'kiev' , 'hotel']
        peopleList = ['edvard', 'employee','ipo','president','firemen','apa','silvia','protectors','','wgo','torsten','juliana','dread','networks','sanjorge','vann','employees','pok','sten','cato','ceo','rebecca','karel','wfa','elian','carman','kapelou','nespola','torsten','trucco','douglas','eggleston','lark','mayor','afghan','philippines','paramurderers','bruno','psychobrotherhood','pakistani','hasidic','brothers','hate','george', 'dombrovski' , 'columbia' ,  'mikhail' , 'Kapolalum ', 'funsho' , 'bukhari' , 'ahmed', 'basra' , 'khouri', 'kasem' , 'leonid', 'nahid', 'otieno', 'owiti', 'leonid' , 'baltasar' , 'hombre' , 'jhon', 'professor' , 'saleh' , 'tanya' , 'mohammed', 'borodinski', 'kashfi' , 'khemkhaeng', 'boonmee' , 'ukrainian' , 'german' , 'italian' , 'dutch', 'french' , 'kapolalum' ,  'funsho' , 'mai', 'korongi', 'lashkar', 'hosain', 'haq', 'maulana', 'bukhari' , 'arab' , 'ali', 'balochi' , 'nicolai' , 'aden'  , 'akram', 'shamsheer' , 'jeddah' , 'kiev', 'abdullah', 'carabobo' , 'bolivar', 'bhutani' , 'jumeirah', 'michieka', 'borodinski', 'otieno', 'wanjohi', 'onyango', 'kenyan', 'nairobi', 'jtomski', 'hakan','vwhombre','jorge','soltan','anka','green','joetomsk','igor','middleman'] 
        for j in range(1,len(finalBag)):    # Each topic
            # print "\n \n \n " , finalBag[j]
           
            # print "topic#: ", j
                
            # ------------------- 2 tokenizer ----------------------
            # print finalBag[j]
            # ------------------- 2 tokenizer ----------------------
            stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') & (word != u'\xe9') & (word != u'\u2018') &(word != u'.\u201d') & (word != u'\u201c') & (word != '\u201c') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
                for document in [finalBag[j]]]
            # stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') &(word != u'.\u201d') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
                # for document in [finalBag[j]]]
            # print "Final stopped", stopped_tokens
            # ------------------- 3 Stemming and Count word frequencies -------------------
            # p_stemmer = PorterStemmer()
            stemmer = {}              
            texts = []	
            texts_set = [] #set()
          
            for stopped_token in stopped_tokens:
                stemmed_texts = [p_stemmer.stem(i) for i in stopped_token]
                texts_set += [stemmed_texts]			
                
            frequency = defaultdict(int)
            for text in texts_set:
                for token in text:
                    frequency[token] += 1
                    
            # Only keep words that appear more than once
            processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts_set]
            # print "\n Final proceesd", processed_corpus
            # ------------------- 4 Dictionary and TF-IDF Vectors -------------------    
            ids2words = my_dictionary.token2id
            bow_corpus = [my_dictionary.doc2bow(text) for text in processed_corpus]
            
            # final_vectors = Text_tfidf[bow_corpus]    # With TF-IDF 	  
            final_vectors = bow_corpus              # No TF-IDF
            
            # print "Input: ", bow_corpus
            # print "Input: ", final_vectors[0]
            new_list = []	
            key_words = []
            # Words from doc TF-IDF Vector
            # Sort word bag of each document 
            if len(final_vectors[0]) > 2: 
                new_list = sorted(final_vectors[0], key=lambda prob: prob[1], reverse=True)
            else:
                new_list = final_vectors[0]

            # print "\n Bag of sorted: ", new_list
            # k = 0;
            # for i in range(0,len(new_list)):   # Pick the firts 10 keywords in sorted list
                # for key in ids2words:
                    # if ids2words[key] == new_list[i][0]: # bow_corpus[1][2][0]:
                        # if (k<10):              # first 3 keywords, no more 
                            # term = de_stemmer[key]
                            # topicWordTags2[j].add(str(term))   # Add this to the bag of words  
                            # k = k+1
            accu = 0
            for each in new_list:
                accu += each[1]
                # print each, each[0], each[1]
            
            for i in range(0,len(new_list)):   # Pick the firts 10 keywords in sorted list
                for key in ids2words:
                    if ids2words[key] == new_list[i][0]: # bow_corpus[1][2][0]:
                        term = de_stemmer[key]
                        if (term in timeList):
                            group = 0
                        elif (term in placeList):
                            group = 1
                        elif (term in peopleList):
                            group = 2
                        else:
                            group = 3
                        score = float(new_list[i][1])/accu
                        # print "score", float(new_list[i][1]), accu, score
                        # if score > 0.01:
                        topicWordTags2[j].append([str(term), group,score])   # Add this to the bag of words  

            print "\n Final Entites: ", len(new_list), topicWordTags2[j]
            
        # ----------------------- Process word tags from document vectors  (to complete 10 minimum tags for each topic)
        for j in range(1,len(doc_topic_keywords)):    # Each topic

            # ------------------- 2 tokenizer ----------------------
            stopped_tokens = [[word for word in WordPunctTokenizer().tokenize(str(document).lower()) if ((word not in stoplist) & (word != u'.\u201d<') & (word != u'\xe9') & (word != u'\u2018') &(word != u'.\u201d') & (word != u'\u201c') & (word != '\u201c') & (len(word) > 2)  & (is_int(word) == False) )]#  & (is_int(word) == False)  & (len(word) > 3) & (len(word) == len(word.strip({0,1,2,3,4,5,6,7,8,9}))) )] #(re.search('\d+',	 word) == False) ) ]
                for document in [doc_topic_keywords[j]]]

            # ------------------- 3 Stemming and Count word frequencies -------------------
            stemmer = {}              
            texts = []	
            texts_set = [] #set()
          
            for stopped_token in stopped_tokens:
                stemmed_texts = [p_stemmer.stem(i) for i in stopped_token]
                texts_set += [stemmed_texts]			
                
            frequency = defaultdict(int)
            for text in texts_set:
                for token in text:
                    frequency[token] += 1
                    
            # Only keep words that appear more than once
            processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts_set]
            # print "\n Final proceesd", processed_corpus
            # ------------------- 4 Dictionary and TF-IDF Vectors -------------------    
            ids2words = my_dictionary.token2id
            bow_corpus = [my_dictionary.doc2bow(text) for text in processed_corpus]
            
            # final_vectors = Text_tfidf[bow_corpus]    # With TF-IDF 	  
            final_vectors = bow_corpus              # No TF-IDF
            
            # print "Input: ", bow_corpus
            # print "Input: ", final_vectors[0]
            new_list = []	
            key_words = []
            # Words from doc TF-IDF Vector
            # Sort word bag of each document 
            if len(final_vectors[0]) > 2: 
                new_list = sorted(final_vectors[0], key=lambda prob: prob[1], reverse=True)
                
            else:
                new_list = final_vectors[0]
            # print "\n Bag of sorted: ", new_list
            k = 0;
            for i in range(0,len(new_list)):   # Pick the firts 10 keywords in sorted list
                for key in ids2words:
                    if ids2words[key] == new_list[i][0]: # bow_corpus[1][2][0]:
                        if (k<20):              # first 3 keywords, no more 
                            term = de_stemmer[key]
                            # topicWordTags3[j].add(str(term))   # Add this to the bag of words  
                            topicWordTags3[j].append([str(term), 3,0.1])   # Add this to the bag of words  
                            k = k+1
            # print "\n Final Entites: ", topicWordTags2[j]
            
        # ------------------------------ Add entities from user interactions ----------------------
        topic_hash = []
        
 
        for i in range(1,class_num+1):    # topicWordTags[0] is always empty,
            tagWords = []
            temp_set = set();
            kk = 0
            for eachTag in topicWordTags2[i]:
                if kk<20:
                    if not (eachTag[0] in temp_set):
                        temp_set.add(eachTag[0])
                        tagWords.append(eachTag)
                        kk = kk+1
    
            # ------------------------------ Add more entities from documents ----------------------- 
            #print topicWordTags3[i]
            if kk<20:
                for eachTag in topicWordTags3[i]:
                    if kk<20: 
                        if not (eachTag[0] in temp_set):
                            temp_set.add(eachTag[0])
                            #print "set > ", temp_set
                            tagWords.append(eachTag)
                            #print "List > ", tagWords
                            kk = kk+1

            tagWords = sorted(tagWords, key=lambda k: k[2],reverse=True)
            temp = {"TopicNum: ": i - 1, "keywords": tagWords}
            topic_hash.append(temp)
           
        
        fout = open(filename3,"w")
        fout.write(json.dumps(topic_hash,indent=1))
        fout.close()

                         
    # print "\n doc_topic_array: ", doc_topic_array	
    # print "\n doc_topic_array: ", doc_key_word	
    return doc_topic_array, doc_key_word

def Read_dataset(json_file):	

    ret = []
    j_ret = []	
	
    all_docs = json.load(open(json_file))
    i = 0
    for a_doc in all_docs:
        new_doc = a_doc["contents"]
        j_ret.append(new_doc)

    return j_ret, len(j_ret)

def Read_user_interactions(interaction_file,docs_number):

    doc_to_text = {}
    from_log_to_id = {}
    doc_counts = {}
    highlight_plus = []
    search_terms = []	
    reading_time = []		
    note_terms = []	
    open_time = []
	
	#--------------------JSON Logs------------------
    for i in xrange(0,docs_number):
        highlight_plus.append("")
        reading_time.append(0)		
        open_time.append(0) 
        

    all_interactions = json.load(open(interaction_file))
    i = 0
    for a_interaction in all_interactions:
    # ------------------- Highlight terms ----------------------				
        if 	a_interaction["InteractionType"] == "Highlight" and a_interaction["ID"] != []:
            #print  a_interaction["ID"]
            if len(a_interaction["ID"].split(" ")) > 1:
                num = int(a_interaction["ID"].split(" ")[1])  - 1
            else: 
                num = 1
            highlight_plus[num] = a_interaction["Text"] + " "

    # ------------------- Reading Time ----------------------			
        if 	a_interaction["InteractionType"] == "Reading" and a_interaction["ID"] != []:
            num = int(a_interaction["ID"].split(" ")[1]) - 1			        
            reading_time[num] += a_interaction["duration"]			

    # ------------------- Open Time ----------------------           
        if  a_interaction["InteractionType"] == "Doc_open" and a_interaction["ID"] != []:
            num = int(a_interaction["ID"].split(" ")[1]) - 1                    
            open_time[num] += a_interaction["duration"]          

    # ------------------- Search terms ----------------------			
        if 	a_interaction["InteractionType"] == "Search":
            search_terms.append(a_interaction["Text"])			

    # ------------------- Notes ----------------------           
        if  a_interaction["InteractionType"] == "Add note":
            note_terms.append(a_interaction["Text"])          
			
    search_list = []	
	
    for i in xrange(0,docs_number):
        search_list.append("")
    # Stop words and tokenize search term in each document

    English_stop_words = get_stop_words('en')
    My_list = ['one','two','three','four','five','six','seven','eight','nine','ten', '://' ,'http', 'www' ,'com',	'are', 'will' ,'said', 'can', 'near', 'and', 'the', 'i', 'a', 'to', 'it', 'was', 'he', 'of', 'in', 'you', 'that', 'but', 'so', 'on', 'up', 'we', 'all', 'for', 'out', 'me', 'him', 'they', 'says', 'got', 'then', 'there', 'no', 'his', 'as', 'with', 'them', 'she', 'said', 'down', 'see', 'had', 'when', 'about', 'what', 'my', 'well', 'if', 'at', 'come', 'would', 'by', 'one', 'do', 'be', 'her', "didn't", 'jim', 'get', "don't", 'time', 'or', 'right', 'could', 'is', 'went', "warn't", "ain't", 'good', 'off', 'over', 'go', 'just', 'way', 'like', 'old', 'around', 'know', 'de', 'now', 'this', 'along', 'en', 'done', 'because', 'back', "it's", 'tom', "couldn't", 'ever', 'why', 'going', 'little', 'some', 'your', 'man', 'never', 'too', 'more', 'say', 'says', 'again', 'how', 'here', 'tell', 'message', 'posted' , 'need' , 'needs' , 'someone', 'government', 'intelligence', 'report']
    
    stoplist_1 = set('a b c d e f g h i j k l m n o p q r s t u v w x y z 1 2 3 4 5 6 7 8 9 0'.split(' ')) # Create a set of enlighs alphabets
    stoplist_2 = set(English_stop_words)	
    stoplist_3 = set('es la . , . <br> <br><br> br > : >< < .< { } [ ] ( ) ,\'\' ` " ? ! - \u201d< \u201d .\u201d \u201d u201d \u2019 \u201c \xe9 !< >!'.split(' ')) # Create a set 
    stoplist_4 = set(My_list)
    
    stoplist = 	stoplist_1 | stoplist_2 | stoplist_3 | stoplist_4
	
    i = 0
    for document in data_set_docs:
        search_list[i] = [word for word in WordPunctTokenizer().tokenize(document.lower()) if ((word in search_terms) & (word not in stoplist) & (word != u'.\u201d<') &(word != u'.\u201d') &(word != u'\u201c') & (len(word) > 2)  & (is_int(word) == False) )]
        i+=1		

    search_terms = []		
    for search in search_list:
        temp = ""	
        for j in xrange(len(search)):		
            temp += search[j] + " "   # Add highlights to the text. // Adding notes to the 		
        search_terms.append(temp)				

    #print doc_plus, "\n", search_terms, "\n",reading_time, "\n",
    #return 0        
    return all_interactions, highlight_plus,reading_time,search_terms,note_terms, open_time

def documents_interaction(magic_number, data_set_docs, highlight_plus,search_terms,note_terms,docs_number, reading_time):  # Highlights, Reading time, search term, note_term
    iter = 0
    newDataset = []	
    reading_weight = []					
    print "Search terms: ", search_terms, "\n"	 		
    print "Highlight terms: ", highlight_plus, "\n"	 		
    print "Reading Time: ", reading_time, "\n"	 			
    tot = sum(list(reading_time))	
    #>>>>print "Total time: " ,tot , "\n"
    
    for each in reading_time:
        reading_weight.append(magic_number* float(each)/tot)
    #>>>>print "\n", "Reading/Total: ", reading_weight   #for x in reading_time #[x/y for x, y in zip(reading_time, sum(list(reading_time)))], "\n"
	
    i = 0	
    for doc in data_set_docs:
        newDataset.append([search_terms[i] + highlight_plus[i] + doc])    # Add highlights to the text. // Adding notes to the 
        i += 1


    #print newDataset 		
    #print "\n size: ", len(newDataset)
    return newDataset, reading_weight

#def classNum(magic_number, doc_name,doc_topic_array,last_class, doc_key_word, reading_time,splitby):
def classNum(Text_lda, magic_number, doc_name,Int_type,Int_text, doc_topic_array,last_class, doc_key_word, reading_time,splitby):

    classNumtoShow = []	
    reading_weight = []					
	
    tot = sum(list(reading_time))	# Create reading_weight from reading time of each document to increase height.
    for each in reading_time:
        reading_weight.append(magic_number* float(each)/tot)
		
    #print "\n my weight: ", reading_weight[149]
	
    if Int_type == "Highlight":  # to fill bag topic wordtags
        thiss = LDA_Topic(Int_type, de_stemmer,[[Int_text]],Text_lda,my_dictionary,Text_tfidf)
        
    if Int_type == "Search":  # To get to topic num and filling topic wordtags
        thiss = LDA_Topic(Int_type, de_stemmer,[[Int_text]],Text_lda,my_dictionary,Text_tfidf)
        #print "\n \n Search: ", thiss, Int_text
        return [thiss], "", 0
    elif Int_type == "Add note":
        thiss = LDA_Topic(Int_type, de_stemmer, [[Int_text]],Text_lda,my_dictionary,Text_tfidf)
        #print "\n \n Note:", thiss, Int_text
        return [thiss], "", 0    
    elif isinstance( doc_name, int ):
        #print "\n", Int_type
        return [last_class], "", 0
    else:		
        if len(doc_name.split(",")) == 2:
            #if doc_name.split(",")[0] != "MyNotes" and doc_name.split(",")[0] != " Prompt Note  " and  doc_name.split(",")[1] != "MyNotes" and doc_name.split(",")[1] != " Prompt Note  " :				
            mystring1 = (doc_name.split(",")[0])
            mystring2 = (doc_name.split(",")[1])
            num1 = int(mystring1.split(splitby)[1]) - 1		
            classNumtoShow.append(doc_topic_array[num1 - 1])	# minus one from topic_array number		
            #            print "\n" , mystring2		
            if ("MyNotes" in mystring2.lower()) or ("note" in mystring2.lower()) or ("prompt" in mystring2.lower()) or ("notes" in mystring2.lower()):    #len(mystring2.split(splitby)[1]) > 5: #len(mystring2.split("y")[1] > 5):#isinstance(mystring2.split("y")[1], int):
                classNumtoShow.append(doc_topic_array[num1 - 1])	# minus one from topic_array number														
            else: 					
                num2 = int(mystring2.split(splitby)[1]) - 1						
                classNumtoShow.append(doc_topic_array[num2 - 1])	# minus one from topic_array number									
            return classNumtoShow, "", 0	
            #else:			
            #    return [last_class]	, "", 0	
        else:
            if (len(doc_name.split(" ")) > 1):
                num = int(doc_name.split(" ")[1]) - 1		
                classNumtoShow.append(doc_topic_array[num - 1])   # minus one from topic_array number
            else:
                 return [last_class], "", 0
                 
            return classNumtoShow , doc_key_word[num - 1] , (1 + reading_weight[num - 1]) 

def seg_duration(Text_lda, magic_number, all_interactions, a_interaction, counter, doc_topic_array, classNumtoShow, last_class, doc_key_word, reading_time, splitby):

    i=0
    still = 1	
    duration_max = a_interaction["duration"]
    time_inter = a_interaction["time"]	
    int_duration = duration_max
	
    #    print "\n time ", time_inter
    #   print "\n duration ",duration_max	

	
    for each_int in all_interactions:
	
        if (still == 1) and (each_int["time"] > time_inter) and (each_int["time"] < (time_inter + duration_max)) and (each_int["InteractionType"] in main_events):
		
            int_duration = each_int["time"] - time_inter - 1 # minues 0.1 seconds
            still = 0 # always stops the duration, below is stop is interaction in the same class occured
            classNumtoShow_2, docKeyWords_2, reading_w_2 = classNum(Text_lda, magic_number, each_int["ID"],each_int["InteractionType"],each_int["Text"], doc_topic_array, last_class, doc_key_word, reading_time, splitby) #reading_weight)			
			#doc_name,Int_type,Int_text
            if (classNumtoShow[0] == classNumtoShow_2[0]):
                still = 0			
            if (len(classNumtoShow_2)>1):
                if (classNumtoShow[0] == classNumtoShow_2[1]):			
                    still = 0				
            if (len(classNumtoShow)>1):
                if (classNumtoShow[1] == classNumtoShow_2[0]):			
                    still = 0			
            if (len(classNumtoShow)>1) and (len(classNumtoShow_2)>1):
                if (classNumtoShow[1] == classNumtoShow_2[1]):			
                    still = 0								
						
    return int_duration
	
def topic_threads(Text_lda, magic_number, all_interactions, doc_topic_array,doc_key_word, docs_number, reading_time, splitby): #reading_weight):

    last_class = 0
    ret = []
    counter = 0
	
    for a_interaction in all_interactions:
        
        classNumtoShow, docKeyWords, reading_w = classNum(Text_lda, magic_number, a_interaction["ID"],a_interaction["InteractionType"],a_interaction["Text"], doc_topic_array, last_class, doc_key_word, reading_time,splitby) #reading_weight)	
        last_class = classNumtoShow[0]		
        int_duration = seg_duration(Text_lda, magic_number, all_interactions, a_interaction, counter, doc_topic_array, classNumtoShow, last_class, doc_key_word, reading_time, splitby)

    # ------------------- Highlight ----------------------				
        if a_interaction["InteractionType"] == "Highlight":
            temp = {"stepHeight": 7 , "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "HighlightTxt", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": [a_interaction["Text"]]}
            ret.append(temp)
    # ------------------- Search ----------------------			
        if a_interaction["InteractionType"] == "Search":
            temp = {"stepHeight": 7, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "search", "ClassNum": classNumtoShow, "DocNum": "",  "tags": [a_interaction["Text"]]} 
            ret.append(temp)
    # ------------------- Connection -------------------
        if a_interaction["InteractionType"] == "Connection":
            temp = {"stepHeight": 5, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "newConnection", "ClassNum": classNumtoShow, "DocNum": a_interaction["Text"],  "tags": [a_interaction["Text"]]} 
            ret.append(temp)			
    # ------------------- Reading -------------------
        #if a_interaction["InteractionType"] == "Reading":
        #    temp = {"stepHeight": , "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "read", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
        #    ret.append(temp)
    # ------------------- Stop ----------------------			
        #    temp = {"UserName": "NO.1", "Time": int(a_interaction["time"] + a_interaction["duration"]),"Duration": int_duration, "InteractionType" : "Stop", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
        #    ret.append(temp)			
    # ------------------- Opening -------------------						
        if a_interaction["InteractionType"] == "Doc_open":
            temp = {"stepHeight": float(5) * reading_w, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "OpenDoc", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
            ret.append(temp)
    # ------------------- Notes ---------------------						
        if a_interaction["InteractionType"] == "CreateNote":
            temp = {"stepHeight": 5, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "createNote", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
            ret.append(temp)			
    # ------------------- Add Notes ----------------------						
        if a_interaction["InteractionType"] == "Add note":
            temp = {"stepHeight": 10, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "addNote", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": a_interaction["Text"]}
            ret.append(temp)						
    # ------------------- Moving ----------------------						
    #    if a_interaction["InteractionType"] == "Draging":
    #        temp = {"stepHeight": 1, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "movingDoc", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
    #        ret.append(temp)						
    # ------------------- Titles ----------------------						
    #    if a_interaction["InteractionType"] == "Mouse_hover":
    #        temp = {"stepHeight": 1, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "titleView", "ClassNum": classNumtoShow, "DocNum": a_interaction["ID"],  "tags": docKeyWords}
    #        ret.append(temp)						
        counter+=1
	
    return ret
	
def topic_interests(Text_lda, class_num, magic_number_2, all_interactions, doc_topic_array,doc_key_word, docs_number, reading_time, splitby):

    last_class = 0
    ret = []
    counter = 0
    #    reading_weight = stepHeight(all_interactions, docs_number)
	
    for a_interaction in all_interactions:
    #    print "\n count: ", counter ," ",magic_number_2, " ",a_interaction["ID"], " ",doc_topic_array, " ",last_class, " ",doc_key_word, " ",reading_time," ",splitby        

        classNumtoShow, docKeyWords, reading_w = classNum(Text_lda, magic_number_2, a_interaction["ID"],a_interaction["InteractionType"],a_interaction["Text"],doc_topic_array, last_class, doc_key_word, reading_time,splitby) #reading_weight)	
    #        classNumtoShow, docKeyWords, reading_w = classNum(magic_number, a_interaction["ID"], doc_topic_array, last_class, doc_key_word, reading_time,splitby) #reading_weight)	        
        last_class = classNumtoShow[0]		
        int_duration = seg_duration(Text_lda, magic_number_2, all_interactions, a_interaction, counter, doc_topic_array, classNumtoShow, last_class, doc_key_word, reading_time, splitby)
        counter+=1        
        for each_class in xrange(1,class_num+1):
            if each_class == classNumtoShow[0]:
                effect = 1
            else:
                effect = -0.3           
    # ------------------- Highlight -----------------				
            if a_interaction["InteractionType"] == "Highlight":
                temp = {"stepHeight": effect*7 , "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "HighlightTxt", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": [a_interaction["Text"]]}
                ret.append(temp)
    # ------------------- Search --------------------			
            if a_interaction["InteractionType"] == "Search":
                temp = {"stepHeight": effect*7, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "search", "ClassNum": [each_class], "DocNum": "",  "tags": [a_interaction["Text"]]} 
                ret.append(temp)
    # ------------------- Connection ----------------
		
    # ------------------- Opening -------------------
            if a_interaction["InteractionType"] == "Doc_open":
                temp = {"stepHeight": effect * float(5) * reading_w, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "OpenDoc", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": docKeyWords}
                ret.append(temp)
    # ------------------- Notes ---------------------						
            if a_interaction["InteractionType"] == "CreateNote":
                temp = {"stepHeight": 5, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "createNote", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": docKeyWords}
                ret.append(temp)			
    # ------------------- Add Notes -----------------						
            if a_interaction["InteractionType"] == "Add note":
                temp = {"stepHeight": effect * 10, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "addNote", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": a_interaction["Text"]}
                ret.append(temp)						
    # ------------------- Moving --------------------						
    #    if a_interaction["InteractionType"] == "Draging":
    #        temp = {"stepHeight": 1, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "movingDoc", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": docKeyWords}
    #        ret.append(temp)						
    # ------------------- Titles --------------------						
    #    if a_interaction["InteractionType"] == "Mouse_hover":
    #        temp = {"stepHeight": 1, "Time": int(a_interaction["time"]),"Duration": int_duration, "InteractionType" : "titleView", "ClassNum": [each_class], "DocNum": a_interaction["ID"],  "tags": docKeyWords}
    #        ret.append(temp)						

	
    return ret
	
def save_threads(obj, filename):
    fout = open(filename,"w")
    fout.write(json.dumps(obj,indent=1))
    fout.close()
    return 0

def segmentation_vector(threads):

    ret = []
    
    #for a_interaction in all_interactions:
  

    return ret

def save_features(data_set_docs, highlight_plus,search_terms, reading_weight, open_time, filename):
    json_hash = []
    count = 0

    for each in reading_time:
        temp = {"docName": name+" "+str(count+1), "content": data_set_docs[count], "reading_time": float(each)/10, "highlight_terms": highlight_plus[count], "search_terms": search_terms[count] , "open_time": float(open_time[count])/10}
        count += 1
        json_hash.append(temp)  

        

    # print "\n", json_hash
    # Write in a .json file
    fout = open(filename,"w")
    fout.write(json.dumps(json_hash,indent=1))
    fout.close()
    return 0

def save_notes(note_terms, saveNotes): 
    json_hash = []
    count = 0

    for each in note_terms:
        temp = {"Note": count, "Text": each}
        count += 1
        json_hash.append(temp)  

        

    # print "\n", json_hash
    # Write in a .json file
    fout = open(saveNotes,"w")
    fout.write(json.dumps(json_hash,indent=1))
    fout.close()
    return 0

# def main():
	
class_num = 1        # Number of topics in LDA 
save_x = 1
save_y = 1
keyword_num = 5       # Number of keyword assigned to each document 
magic_number = 20     # How much reading time interaction should effect document vectors (was 40)
magic_number_2 = 10   # How much Doc_open time intetaction should effect ProvThread steps  (was 40)
new_model = 0         # 1 = Yes / 0 = No
LDA_passes = 10 
EntList = 1; # Generate Entities list  

# saveContext()
# restoreContext()

for class_num in xrange(3,4):   #(1,11):

    for dataset_num in xrange(1,4):        #(1,4):
        
        if dataset_num == 1:
            splitby = "g"
            dataset = "Arms"	
            name = "Armsdealing"		
        if dataset_num == 2:
            splitby = "y"
            dataset = "Terrorist"
            name = "TerroristActivity"				
        if dataset_num == 3:
            splitby = "ce"		
            dataset = "Disappearance"
            name = "Disappearance"				
		
        for participant_num in xrange(1,9):  #(1,9):
            print "\n \n Dataset number:", dataset_num , "P Nuumber: ", participant_num, "Class_Num: ", class_num	
            
            # Read documents and user interaction logs
            interactionFile = "./Dataset_"+str(dataset_num)+"/UserInteractions/"+str(dataset)+"_P"+str(participant_num)+"_InteractionsLogs.json";
            datasetFile = "./Dataset_"+str(dataset_num)+"/Dataset/Dataset_"+str(dataset_num)+".json"
            data_set_docs, docs_number = Read_dataset(datasetFile)
            all_interactions, highlight_plus,reading_time,search_terms,note_terms, open_time= Read_user_interactions(interactionFile,docs_number) 
            
            # document vectors
            document_plus, reading_weight = documents_interaction(magic_number, data_set_docs, highlight_plus,search_terms,note_terms,docs_number, reading_time)    

            saveFeatures = "./Dataset_"+str(dataset_num)+"/Dataset/Dataset_"+str(dataset_num)+"_features_P"+str(participant_num)+".json"
            saveNotes = "./Dataset_"+str(dataset_num)+"/Dataset/Notes_P"+str(participant_num)+".json"
            save_features(data_set_docs, highlight_plus,search_terms, reading_weight, open_time, saveFeatures)
            save_notes(note_terms, saveNotes)

            # # Run LDA, save results for each document
            # finalBag, topicWordTags,topicWordTags2,topicWordTags3,de_stemmer, ids2words_, doc_vectors_, Text_lda_, my_dictionary,Text_tfidf, output_topics_, de_stemmer_, doc_topics_ = LDA_Topic_Clustering(document_plus,reading_weight, new_model ,class_num , LDA_passes,dataset_num, participant_num)

            # # Extract and save document keywords
            # docTopicsFile = "./TopicDocs/" +str(dataset) + "_P" + str(participant_num) + "_ClassNum" + str(class_num) + ".json"
            # ldaTopicsFile = "./Dataset_" +str(dataset_num)+ "/LDATopics/" + str(dataset) + "_P" + str(participant_num) + "_ClassNum" + str(class_num) + ".json"
            # entityListFile = "./Dataset_" +str(dataset_num)+ "/EntitiesList/" + str(dataset) + "_P" + str(participant_num) + "_ClassNum" + str(class_num) + ".json"
            # doc_topic_array_, doc_key_word_ = save_topic_docs(0,my_dictionary, docs_number, ids2words_, doc_vectors_, output_topics_, doc_topics_,de_stemmer_, class_num,keyword_num, docTopicsFile, ldaTopicsFile, entityListFile)
 
            # # calculate, sort and save topic threads
            # threads = topic_threads(Text_lda_, magic_number_2, all_interactions, doc_topic_array_, doc_key_word_, docs_number, reading_time, splitby) 
            # threads = sorted(threads, key=lambda k: k['Time'])
            # save_threads(threads, "./Dataset_" + str(dataset_num) + "/ProvThreads/" + str(dataset) +"_P" + str(participant_num) + "_topicThread_"+ str(class_num)+ ".json" )
    
            # # calculate, sort and save interests threads
            # interests = topic_interests(Text_lda_,class_num, magic_number_2, all_interactions, doc_topic_array_, doc_key_word_, docs_number, reading_time, splitby)
            # interests = sorted(interests, key=lambda k: k['Time'])
            # save_threads(interests, "./Dataset_" + str(dataset_num) + "/ProvThreads/" + str(dataset) +"_P" + str(participant_num) + "_topicInterest_" + str(class_num)+ ".json" )

            # doc_topic_array_, doc_key_word_ = save_topic_docs(1,my_dictionary, docs_number, ids2words_, doc_vectors_, output_topics_, doc_topics_,de_stemmer_, class_num,keyword_num, docTopicsFile, ldaTopicsFile, entityListFile)

            # print "total time: ", datetime.now() - lastTime
            # lastTime = datetime.now()
    

print "\n"
print "total time: ", datetime.now() - startTime
print "End"



