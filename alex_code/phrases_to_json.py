import sys
import json
import numpy as np
from sklearn import manifold, decomposition, preprocessing, cluster
import operator

if len(sys.argv)<5:
    print("<phrases input text file> <doc-topics input text file> <output topics-terms json file> <output clusters-terms json file> <output documents json file> <num clusters>")
    exit()

terms_doc = sys.argv[1]
doc_topics_doc = sys.argv[2]
topic_terms_json = sys.argv[3]
clusters_terms_json = sys.argv[4]
documents_json = sys.argv[5]
num_clusters = int(sys.argv[6])

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
            #print(n,return_terms,remove_idxs)
    
        """
        for i,term in enumerate(sorted_terms[-1*n:]):
            contained_in_other_term = False
            for rt in return_terms:
                if term[0] in rt:
                    contained_in_other_term = True
                    break
            if not contained_in_other_term:
                return_terms.append(term[0])
            if len(return_terms)==n:
                    break#reached number of terms
        """
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
    doc_json.append(d)
with open(documents_json, 'w') as outfile:
    json.dump(doc_json, outfile, indent=4, sort_keys=True)

#topic json
topic_json = []
for topic in topic_terms:
    t={}
    doc_row = [0.0 for topic in topic_terms] #fake
    doc_row[topic] = 1.0 #only use from one topic
    t["keywords"] = get_top_n_terms(doc_row,topic_terms,n=15)
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

