import sys
import os
import subprocess
import glob

datasets = ["Arms","Terrorist","Dissapearance"]

#Converting Text Corpus
subprocess.call(["python","create_corpus_from_txt.py","corpus_json","corpus_text"])

def delete_files_in_dir(path):#path ends in folder/*
    files = glob.glob(path)
    for f in files:
        os.remove(f)

#Running LDA (just once per dataset)
NUM_TOPICS = int(sys.argv[1])
for i,dataset in enumerate(datasets):
    #Run LDA
    subprocess.call(["bash","run_lda.sh",
                     os.path.abspath('corpus_text/documents_'+str(i+1)),
                     os.path.abspath('mallet_results/' + dataset),
                     str(NUM_TOPICS)])
                     
    delete_files_in_dir('results_json/'+dataset+"/topics/*")
    delete_files_in_dir('results_json/' + dataset + "/tasks/*")
    delete_files_in_dir('results_json/' + dataset + "/documents/*") 
    
    #Get JSON from LDA for num_tasks clusters
    for num_tasks in range(3,11):
        subprocess.call(["python","phrases_to_json.py",
                        os.path.abspath('mallet_results/' + dataset +"/terms.txt"),
                        os.path.abspath('mallet_results/' + dataset + "/doc_topics.txt"),
                        os.path.abspath('results_json/'+dataset+"/topics/"+dataset+"_"+str(NUM_TOPICS)+".json"),
                        os.path.abspath('results_json/' + dataset + "/tasks/" + dataset + "_" + str(num_tasks) + ".json"),
                        os.path.abspath('results_json/' + dataset + "/documents/" + dataset + "_" + str(num_tasks) + ".json"),
                        str(num_tasks)])

print("-- DONE --")
