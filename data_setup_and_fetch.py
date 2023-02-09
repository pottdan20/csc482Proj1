import os, random
import nltk
from urllib import request 
from zipfile import ZipFile
from bs4 import BeautifulSoup   

# test   

def fetch_part1_document_training():
    trainings = []
    if os.path.exists("part1/document/training"):
        for filename in os.listdir("part1/document/training"): 
            html = open(os.path.join("part1/document/training", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings

def fetch_part1_document_testing():
    trainings = []
    if os.path.exists("part1/document/testing"):
        for filename in os.listdir("part1/document/testing"): 
            html = open(os.path.join("part1/document/testing", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings 

def fetch_part1_paragraph_training():
    trainings = []
    if os.path.exists("part1/paragraph/training"):
        for filename in os.listdir("part1/paragraph/training"): 
            html = open(os.path.join("part1/paragraph/training", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings

def fetch_part1_paragraph_testing():
    trainings = []
    if os.path.exists("part1/paragraph/testing"):
        for filename in os.listdir("part1/paragraph/testing"): 
            html = open(os.path.join("part1/paragraph/testing", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings

def fetch_part1_sentence_testing():
    trainings = []
    if os.path.exists("part1/sentence/testing"):
        for filename in os.listdir("part1/sentence/testing"): 
            html = open(os.path.join("part1/sentence/testing", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings

def fetch_part1_sentence_training():
    trainings = []
    if os.path.exists("part1/sentence/training"):
        for filename in os.listdir("part1/sentence/training"): 
            html = open(os.path.join("part1/sentence/training", filename)).read()
            raw = BeautifulSoup(html, 'html.parser').get_text()
            trainings.append([filename,raw])
    else:
        return None
    return trainings

def write_to_folder( docs):
    for doc in docs:
        f = open(os.path.join("./", doc[0]),'x')
        f.write(doc[1]) #saves as filename and saves raw content
        f.close()

def separate_training(docs):
    random.shuffle(docs)
    size = len(docs)
    i = int(0.2 * size)
    testing = docs[:i]
    training = docs[i:]
    return [testing, training]

def part2(documents):
    authors_in_1 = set()
    half_1 = []
    half_2 = []
    random.shuffle(documents)
    for d in documents:
        a_id = d[0].split("_")[1]
        if a_id not in authors_in_1:
            half_1.append(d)
            authors_in_1.add(a_id)
        else:
            half_2.append(d)
    doc_training = half_1
    random.shuffle(half_2)
    i = int(len(half_2) * 0.4)
    doc_testing = half_2[:i] # 0.4 of half is 0.2 of whole dataset

    #save in folders to separate data
    os.mkdir('./part2')
    os.chdir("./part2")
    os.mkdir("./document")
    os.chdir("./document")
    os.mkdir("./testing")
    os.chdir("./testing")
    write_to_folder(doc_testing) #write testing to testing folder
    os.chdir("../")
    os.mkdir("./training")
    os.chdir("./training")
    write_to_folder(doc_training)
    os.chdir("../../../")


    #now working on paragraphs per author
    authors = {}
    par_testing = []
    par_training = []
    random.shuffle(documents)
    x = 0
    for d in documents:
        a_id = d[0].split("_")[1]
        if a_id not in authors.keys(): #maps each paragraph to an author
            authors[a_id] = []
        authors[a_id].append([str(x) + "-" + d[0], d[1]])
        x += 1

    for a in authors.keys(): # 80:20 split of each authors writing
        i = int(0.8 * len(authors[a]))
        random.shuffle(authors[a])
        for p in authors[a][:i]:
            par_training.append(p)
        for p in authors[a][i:]:
            par_testing.append(p)
    

    os.chdir("./part2")
    os.mkdir("./paragraph")
    os.chdir("./paragraph")
    os.mkdir("./testing")
    os.chdir("./testing")
    write_to_folder(par_testing) #write testing to testing folder
    os.chdir("../")
    os.mkdir("./training")
    os.chdir("./training")
    write_to_folder(par_training)
    os.chdir("../../../")
    # save in folder




def fetch_and_separate_original_data():
    zip_url = "http://users.csc.calpoly.edu/~foaad/proj1F21_files.zip"
    # Download zip file from url
    zip_response = request.urlopen(zip_url)
    # Create a new file 
    temp_zip = open("./tempfile.zip", "wb")
    # Write contents of downloaded zip into new file 
    contents = zip_response.read()
    temp_zip.write(contents)
    temp_zip.close()
    # Re-open and extract contents into current directory
    in_zip_file = ZipFile("./tempfile.zip")
    in_zip_file.extractall(path = "./in_files_html")
    os.remove("./tempfile.zip") # remove the zip file
    in_zip_file.close()
    # Go to directory within created directory (proj1F21_files)
    os.chdir("./in_files_html") # go to in_files_html directory 
    child_directory = os.listdir()[0] # cd to directory inside it 
    os.chdir("./" + child_directory)
    # Look at each html file 
    documents = []
    for filename in os.listdir(): 
        html = open(filename).read()
        raw = BeautifulSoup(html, 'html.parser').get_text()
        # PROCESS RAW HERE (label and do whatever)
        
        documents.append([filename[:-4] + "txt",raw])
        os.remove(filename) # remove file after done processing


    os.chdir("..")
    os.rmdir("./" + os.listdir()[0])
    os.chdir("..")
    os.rmdir("./in_files_html")   



    # testingAndTraining = separate_training(documents) # returns [testing, training]
    # os.mkdir('./part1')
    # os.chdir("./part1")
    # os.mkdir("./document")
    # os.chdir("./document")
    # os.mkdir("./testing")
    # os.chdir("./testing")
    # write_to_folder(testingAndTraining[0]) #write testing to testing folder
    # os.chdir("../")
    # os.mkdir("./training")
    # os.chdir("./training")
    # write_to_folder( testingAndTraining[1])
    # os.chdir("../../../")

    # paragraphs = [] #separating paragraphs for testing and training data
    # x = 0
    # for doc in documents:
    #     ps = doc[1].split('\n')
    #     for p in ps:
    #         if len(p) < 2:
    #             ps.remove(p)
    #         else:
    #             paragraphs.append([str(x) +"-"+ doc[0], p])
    #             x += 1
    
    # testingAndTraining = separate_training(paragraphs)
    
    # os.chdir("./part1")
    # os.mkdir("./paragraph")
    # os.chdir("./paragraph")
    # os.mkdir("./testing")
    # os.chdir("./testing")
    # write_to_folder(testingAndTraining[0]) #write testing to testing folder
    # os.chdir("../")
    # os.mkdir("./training")
    # os.chdir("./training")
    # write_to_folder(testingAndTraining[1])
    # os.chdir("../../../")


    # sents = [] #separating sents for testing and training data
    # x = 0
    # for doc in documents:
    #     tempSents = nltk.sent_tokenize(doc[1])
    #     for s in tempSents:
    #         if len(s) < 2:
    #             tempSents.remove(p)
    #         else:
    #             sents.append([str(x) +"-"+ doc[0], s])
    #             x += 1
    
    # testingAndTraining = separate_training(sents)
    
    # os.chdir("./part1")
    # os.mkdir("./sentence")
    # os.chdir("./sentence")
    # os.mkdir("./testing")
    # os.chdir("./testing")
    # write_to_folder(testingAndTraining[0]) #write testing to testing folder
    # os.chdir("../")
    # os.mkdir("./training")
    # os.chdir("./training")
    # write_to_folder(testingAndTraining[1])
    # os.chdir("../../../")
    
    part2(documents)


        
fetch_and_separate_original_data()

# doc_train = fetch_part1_document_training()
# par_train = fetch_part1_paragraph_training()
# sent_train = fetch_part1_sentence_training()

# print(sent_train[10])
# print(sent_train[9])