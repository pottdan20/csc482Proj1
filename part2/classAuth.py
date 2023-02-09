import sys
import re
import nltk
import cmudict
import syllables
import math 
import os
import random

cmu = cmudict.dict()

class DataPoint:
    def __init__(self, label, raw):
        self.label = label
        self.raw = raw

        paragraphs=raw.split("\n\n")
        sentences = nltk.tokenize.sent_tokenize(raw)
        words = re.sub(r'[^\w\s]', '', raw).split()
        syllableList = [syllableCount(w) for w in words]

        syCount=0
        gunningCount=0
        for sy in syllableList:
            if sy>3:
                syCount+=1

            if sy>2:
                gunningCount+=1
                
        nosw=(syCount/len(syllableList))*100  
        PHW = gunningCount/len(syllableList)
        ASL=len(words)/len(sentences)


        self.features = {}
        self.features["ASL"]=round(ASL) #Average Sentence Length
        self.features["APL"]=round(len(sentences)/len(paragraphs)) #Average Paragraph Length
        
        self.features["AWC"]=round(nosw)
        
        self.features["commasPerSentence"] = round((raw.count(",")/len(sentences))*10)
        #self.features["colonsPerSentence"] = raw.count(":")/len(sentences)
        #self.features["semiPerSentence"] = raw.count(";")/len(sentences)

        self.features["lexicalComplexity"]= round((len(list(set(words)))/len(words))*10)

        self.features["FleshScore"] = round(1.599*nosw - 1.015*ASL - 31.517)
        self.features["FleshKincade"] = round(0.39*ASL + 11.8*nosw - 15.59)
        self.features["gunnongFog"] = round(0.4*(ASL + PHW))
     
    def __repr__(self):
        return("Label: " + str(self.label) + 
        "\nFeatures: " + str(self.features) )
    
def syllableCount(word):
    cmuWord = cmu.get(word)
    if cmuWord == None:
        return syllables.estimate(word)

    return len([1 for s in cmuWord[0] if s[-1].isdigit()])

def setUp():
    documentsTraining=[]
    documentsTesting=[]
    paragraphsTraining=[]
    paragraphsTesting=[]
    
    os.chdir("./part2/document/training")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw)
        documentsTraining.append((dp.features,dp.label))

    os.chdir("../../document/testing")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw)
        documentsTesting.append((dp.features,dp.label))

    os.chdir("../../paragraph/training")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw)
        paragraphsTraining.append((dp.features,dp.label))

    os.chdir("../../paragraph/testing")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw)
        paragraphsTesting.append((dp.features,dp.label))
    
    
    print("Document Accuracy:" + str(naiveBayes(documentsTraining, documentsTesting) ))
    print("Paragraph Accuracy:" + str(naiveBayes(paragraphsTraining, paragraphsTesting)))
    


    os.chdir("../../")
    t = open("right.txt", "w")
    f = open("wrong.txt", "w")
    g = open("dataPoints.txt" , "w")
    for all in documentsTraining:
        g.write(str(all)+ " \n")
    g.close()
    classifier = nltk.NaiveBayesClassifier.train(documentsTraining)
    for (name, tag) in documentsTesting:
        
        guess = classifier.classify(name)
        
        if guess != tag:
            #f.write((tag, guess, name))
            f.write(str(tag) + " " + str(guess) + " " + str(name) + "\n")
        else:
            t.write(str(tag) + " " + str(guess) + " " + str(name)+ "\n")
    #print(errors)
    f.close()
    t.close()
def naiveBayes(train_set,test_set):
    #print(train_set)
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    return(nltk.classify.accuracy(classifier, test_set))


setUp()