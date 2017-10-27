import sys
import json
import numpy as np
from sklearn import manifold, decomposition, preprocessing, cluster
import operator
from bs4 import BeautifulSoup
import os
import subprocess
import copy
import glob

NUM_KEYWORDS_TOPIC=15 #number of keywords to display per TOPIC
NUM_KEYWORDS_TASK=15 #number of keywords to display per TASK

#Weighting for Participant File Output
field_weights = {}
field_weights["reading_time"] =     1.00  #(this * duration * topic weight) + topic weight
field_weights["open_time"] =        0.25  #(this * duration * topic weight) + topic weight
field_weights["search_terms"] =     3.00  #multiply topic weight against field
field_weights["highlight_terms"] =  3.00  #multiply topic weight against field

if len(sys.argv)<5:
    print("<phrases input text file> <doc-topics input text file> <output topics-terms json file> <output clusters-terms json file> <output documents json file> <participant data dir> <participant out directory> <mallet results dir> <num clusters>")
    exit()

terms_doc = sys.argv[1]
doc_topics_doc = sys.argv[2]
topic_terms_json = sys.argv[3]
clusters_terms_json = sys.argv[4]
documents_json = sys.argv[5]
participant_data_dir = sys.argv[6]
partipant_out_dir = sys.argv[7]
mallet_results_dir = sys.argv[8]
num_clusters = int(sys.argv[9])

topic_terms = {} #topic to term to weight
with open(terms_doc) as infile:
    topic = -1
    for line in infile:
        if line.startswith("---Topic"):
            topic += 1
            topic_terms[topic] = {}
            continue
        else:
            split = line.split("\t")
            if len(split)<2:
                continue
            topic_terms[topic][split[0]]=float(split[1])

docs = []
doc_topics = []
with open(doc_topics_doc) as infile:
    for line in infile:
        split = line.split()

        name = split[1].split('/')[-1]
        docs.append(name)
        doc_topics.append([float(s) for s in split[2:]])

minmax = preprocessing.MinMaxScaler()
data = minmax.fit_transform(doc_topics)
mu = np.mean(data, axis=0)
pca = decomposition.PCA(2)
data = pca.fit_transform(data)
clusterer = cluster.KMeans(num_clusters)
labels = clusterer.fit_predict(data)

#returns top n terms per document
def get_top_n_terms(doc_row,topic_terms,n=5,remove_subterms=True):
    terms = {}
    for i,v in enumerate(doc_row):
        for term,w in topic_terms[i].items():
            if term not in terms:
                terms[term] = 0.0
            terms[term] += w*v
    sorted_terms = sorted(terms.items(), key=operator.itemgetter(1))
    return_terms = []
    rcount=0
    if remove_subterms:
        iterations=0
        while len(return_terms)<n and iterations<100:
            iterations+=1
            for i,term in enumerate(list(reversed(sorted_terms))[rcount:rcount+n]):
                return_terms.append(term)
                if len(return_terms)==n:
                    rcount += 1+i
                    break
            #print(n,return_terms)
            remove_idxs=[]
            return_terms = sorted(return_terms, key=lambda x:-1*len(x[0]))
            #print([term[0] for term in return_terms])
            for i in range(0,len(return_terms)):
                for j in range(i+1,len(return_terms)):
                    if return_terms[j][0] in return_terms[i][0]:
                        remove_idxs.append(j)
                        break
            old_return_terms = return_terms
            return_terms = []
            for i,term in enumerate(old_return_terms):
                if i not in remove_idxs:
                    return_terms.append(term)
        return_terms = [term[0] for term in sorted(return_terms, key=lambda x:x[1])][0:n]
    else:
        return_terms = [term[0] for term in sorted_terms[-1*n:]]
    return return_terms

#document json
doc_json = []
for i,doc in enumerate(docs):
    d = {}
    d["docName"] = doc
    d["classNum"] = [[int(labels[i]),1.0]]#NO CONFIDENCE VALUE RIGHT NOW
    d["events"]=get_top_n_terms(doc_topics[i],topic_terms)
    d["topicWeights"]=doc_topics[i]
    doc_json.append(d)
with open(documents_json, 'w') as outfile:
    json.dump(doc_json, outfile, indent=4, sort_keys=True)

#topic json
topic_json = []
for topic in topic_terms:
    t={}
    doc_row = [0.0 for topic in topic_terms] #fake (using zeros for non-topic)
    doc_row[topic] = 1.0 #only use from one topic
    t["keywords"] = get_top_n_terms(doc_row,topic_terms,n=NUM_KEYWORDS_TOPIC)
    t["TopicNum"] = topic
    topic_json.append(t)
with open(topic_terms_json, 'w') as outfile:
    json.dump(topic_json, outfile, indent=4, sort_keys=True)

#clusters json
cluster_json = []
for i,center in enumerate(clusterer.cluster_centers_):
    c={}
    #reconstructing from cluster center and PCA model
    doc_row = np.dot(center,pca.components_)
    doc_row+=mu
    doc_row = minmax.inverse_transform(doc_row)

    c["TopicNum"] = i
    c["keywords"] = get_top_n_terms(doc_row,topic_terms,n=15)
    cluster_json.append(c)
with open(clusters_terms_json, 'w') as outfile:
    json.dump(cluster_json, outfile, indent=4, sort_keys=True)

#Participant only stuff (weight shown terms and classify their notes)

#Read in each participant interaction json file
file_data = {}
file_notes = {}
for filename in os.listdir(participant_data_dir):
    if "P" not in filename:
        continue
    if "Note" in filename:
        with open(os.path.join(participant_data_dir,filename),'r') as jf:
            file_notes[filename] = json.load(jf)
        continue
    with open(os.path.join(participant_data_dir,filename),'r') as jf:
        file_data[filename] = json.load(jf)

#Write Notes now with Topic Weights and Cluster Assignment
def delete_files_in_dir(path):#path ends in folder/*
    files = glob.glob(path)
    for f in files:
        os.remove(f)
for file in file_notes.keys():
    participant = file.split("_")[-1].split(".")[0]
    participant_json = []
    for nidx,note_json in enumerate(file_notes[file]):
        delete_files_in_dir('tmp/*')
        with open("tmp/ptext.txt",'w') as t:
            #print("\t{}".format(note_json))
            #Get Topics
            t.write(note_json["Text"])
        subprocess.call(["bash","predict_topics.sh",
                                 mallet_results_dir,
                                 os.path.join(os.getcwd(),"tmp","ptext.txt"),
                                 os.path.join(os.getcwd(),"tmp","topic.txt")
                             ])
        with open("tmp/topic.txt",'r') as tt:
            text_topics = tt.readlines()
            if len(text_topics)<=1:
                continue
            text_topics = text_topics[1]
            topics = [float(i) for i in text_topics.split()[2:]]
            file_notes[file][nidx]["Topics"] = topics
            #get cluster
            topics = minmax.transform(topics)
            data = pca.transform(topics)
            label = clusterer.predict(data)
            file_notes[file][nidx]["Task"] = int(label[0])
    #Write to File
    if not os.path.exists(os.path.join(partipant_out_dir,participant)):
        os.mkdir(os.path.join(partipant_out_dir,participant))
    notes_path = os.path.join(partipant_out_dir,participant,"notes_"+str(num_clusters)+".json")
    with open(notes_path,'w') as outfile:
         json.dump(file_notes[file], outfile, indent=4, sort_keys=True)


for file in file_data.keys():
    participant = file.split("_")[-1].split(".")[0]
    #print(participant)#
    data = file_data[file]
    p_topic_terms = copy.deepcopy(topic_terms);#will modify
    p_doc_topics = copy.deepcopy(doc_topics);#will modify
    for i,doc in enumerate(data):
        for topic in p_topic_terms.keys():
            for term in p_topic_terms[topic].keys():
                #multiplier fields
                for field in ["search_terms","highlight_terms"]:
                    for t in doc[field].split():
                        if t.lower() in term.lower():
                            p_topic_terms[topic][term] *= field_weights[field]
                #additive fields
                for field in ["reading_time","open_time"]:
                    for j,dt in enumerate(doc_topics[i]):
                        p_doc_topics[i][j] += dt*field_weights[field];

        #Saving Files using new biased weights
        documents_json = os.path.join(partipant_out_dir,participant,"documents_"+str(num_clusters)+".json")
        topic_terms_json = os.path.join(partipant_out_dir,participant,"topics_"+str(num_clusters)+".json")
        clusters_terms_json = os.path.join(partipant_out_dir,participant,"tasks_"+str(num_clusters)+".json")
        
        #document json
        doc_json = []
        for i,doc in enumerate(docs):
            d = {}
            d["docName"] = doc
            d["classNum"] = [[int(labels[i]),1.0]]#NO CONFIDENCE VALUE RIGHT NOW
            d["events"]=get_top_n_terms(p_doc_topics[i],p_topic_terms)
            d["topicWeights"]=p_doc_topics[i]
            doc_json.append(d)
        with open(documents_json, 'w') as outfile:
            json.dump(doc_json, outfile, indent=4, sort_keys=True)

        #topic json
        topic_json = []
        for topic in topic_terms:
            t={}
            doc_row = [0.0 for topic in p_topic_terms] #fake (using zeros for non-topic)
            doc_row[topic] = 1.0 #only use from one topic
            t["keywords"] = get_top_n_terms(doc_row,p_topic_terms,n=NUM_KEYWORDS_TOPIC)
            t["TopicNum"] = topic
            topic_json.append(t)
        with open(topic_terms_json, 'w') as outfile:
            json.dump(topic_json, outfile, indent=4, sort_keys=True)

        #clusters json
        cluster_json = []
        for i,center in enumerate(clusterer.cluster_centers_):
            c={}
            #reconstructing from cluster center and PCA model
            doc_row = np.dot(center,pca.components_)
            doc_row+=mu
            doc_row = minmax.inverse_transform(doc_row)
            
            c["TopicNum"] = i
            c["keywords"] = get_top_n_terms(doc_row,p_topic_terms,n=NUM_KEYWORDS_TASK)
            cluster_json.append(c)
        with open(clusters_terms_json, 'w') as outfile:
            json.dump(cluster_json, outfile, indent=4, sort_keys=True)






