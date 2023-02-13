import sys
import re
import nltk
import cmudict
import syllables
import math 
import os
import random
import data_setup_and_fetch 

cmu = cmudict.dict()
with open("./part2/DaleChallEasyWordList.txt") as f:
    daleList = f.read().lower().split()

class DataPoint:
    def __init__(self, label, raw, word_features):
        self.label = label
        self.raw = raw
        self.features = {}
        self.word_features = word_features

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

        daleCount=0 
        for word in words:
            if word not in daleList:
                daleCount+=1
        PDW = daleCount/len(words)
        
        if PDW < 0.05:
            self.features["Dale"]= round((0.1579*PDW + 0.0496*ASL)*10)
        else:
            self.features["Dale"]=  round((0.1579*PDW + 0.0496*ASL + 3.6365)*10)
        
       
        #self.features["ASL"]=round(ASL) #Average Sentence Length
        self.features["APL"]=round(len(sentences)/len(paragraphs)) #Average Paragraph Length
        
        self.features["AWC"]=round(nosw)
        
        self.features["commasPerSentence"] = round((raw.count(",")/len(sentences))*10)
        #self.features["colonsPerSentence"] = raw.count(":")/len(sentences)
        #self.features["semiPerSentence"] = raw.count(";")/len(sentences)

        self.features["lexicalComplexity"]= round((len(list(set(words)))/len(words))*10)

        self.features["FleshScore"] = -1*round(1.599*nosw - 1.015*ASL - 31.517)
        self.features["FleshKincade"] = round(0.39*ASL + 11.8*nosw - 15.59)
        self.features["gunningFog"] = round(0.4*(ASL + PHW))
        #self.features["SMOG"] = round(0.4*(ASL + (gunningCount/len(syllableList))))
        
        # Common words as features 
        # for common_word in word_features: 
        #     if common_word in set(words): 
        #         self.features[common_word] = True 
        #     else: 
        #         self.features[common_word] = False 
        
        # Parts of Speech as features 
        # pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS', 'CC', 'MD']
        # tagged_text = nltk.pos_tag(words)
        # fd = nltk.FreqDist(tag for (word, tag) in tagged_text)
        # for pos in pos_list: 
        #     self.features[pos] = fd.get(pos)
     
    def __repr__(self):
        return("Label: " + str(self.label) + 
        "\nFeatures: " + str(self.features) )
    
def syllableCount(word):
    cmuWord = cmu.get(word)
    if cmuWord == None:
        return syllables.estimate(word)

    return len([1 for s in cmuWord[0] if s[-1].isdigit()])

def setUp():
    word_features = get_word_features()
    documentsTraining=[]
    documentsTesting=[]
    paragraphsTraining=[]
    paragraphsTesting=[]
    
    os.chdir("./part2/document/training")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw, word_features)
        documentsTraining.append((dp.features,dp.label))

    os.chdir("../../document/testing")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw, word_features)
        documentsTesting.append((dp.features,dp.label))

    os.chdir("../../paragraph/training")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw, word_features)
        paragraphsTraining.append((dp.features,dp.label))

    os.chdir("../../paragraph/testing")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        dp=DataPoint(file.split("_")[1], docRaw, word_features)
        paragraphsTesting.append((dp.features,dp.label))
    
    
    print("Document Accuracy:" + str(naiveBayes(documentsTraining, documentsTesting) ))
    print("Paragraph Accuracy:" + str(naiveBayes(paragraphsTraining, paragraphsTesting)))
    #k_fold(documentsTraining)
    #k_fold(paragraphsTraining)
    


    os.chdir("../../")
    t = open("right.txt", "w")
    f = open("wrong.txt", "w")
    g = open("dataPoints.txt" , "w")
    for all in paragraphsTraining:
        g.write(str(all)+ " \n")
    g.close()
    classifier = nltk.NaiveBayesClassifier.train(paragraphsTraining)
    for (name, tag) in paragraphsTesting:
        
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

# Returns the error of classifier on test_set 
# The error is just the number of incorrect classifications 
def get_error(test_set, classifier):
    golden_labels = []
    predicted_labels = []
    for data_element in test_set:
        golden_labels.append(data_element[1]) # label 
        predicted_labels.append(classifier.classify(data_element[0])) # predict label for element
    error = 0
    for i in range(len(golden_labels)): 
        if golden_labels[i] != predicted_labels[i]: 
            error += 1
    return error

# Returns the avg. error through 10-fold cross validation for each model 
# training_data can be for documents, paragraphs, or sentences 
def k_fold(training_data): 
    training_set = training_data
    random.shuffle(training_set)
    nb_error = 0
    index = 0
    fold_length = len(training_set)//10 # Takes 10 folds, rounds down length (because it may not be possible to get folds of same length)
    for k in range(10): 
        training_folds = []
        testing_fold = []
        if k == 9: 
            # Getting the last fold, just take the end of the training list 
            testing_fold = training_set[index:]
            training_folds = training_set[0:index]
        else:                                 
            # Getting a fold at the beginning of the list
            testing_fold = training_set[index:index+fold_length]
            training_folds = training_set[0:index] + training_set[index+fold_length:]
            index += fold_length  
        # Get error for each model and sum them 
        nb_classifier = nltk.NaiveBayesClassifier.train(training_folds)
        nb_error += get_error(testing_fold, nb_classifier)   
    # Average the errors and display
    nb_error = nb_error / 10
    print("Naive Bayes Avg. Error: " + str(nb_error))

def get_word_features(): 
    word_features = set()
    training_documents = data_setup_and_fetch.fetch_part2_document_training()
    all_words = []    
    # Get all words from training documents, and use most common 500 as features 
    for doc in training_documents:
        raw_text = doc[1]
        text = re.sub(r"[^a-zA-Z\s\-]", "", raw_text) # Remove punctuation and numbers (but keep -)
        words = [w for w in nltk.word_tokenize(text)]
        all_words += words            
    freq_dist = nltk.FreqDist(all_words)         
    common_words = freq_dist.most_common(500) 
    for (word, count) in common_words: 
        word_features.add(word)
    return word_features
    
setUp()