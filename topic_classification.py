import nltk
import data_setup_and_fetch
import re 
import random 
from nltk.stem.porter import PorterStemmer 

class TopicClassifier:
    def __init__(self):
        self.document_classifier = None
        self.paragraph_classifier = None
        self.sentence_classifier = None
        self.word_features = set()
        self.stemmer = PorterStemmer()
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.k = 10
        self.set_features()
    
    
    # Does preprocessing and tokenization of text
    # Returns a list of tokens 
    def get_tokens(self, text): 
        raw_text = re.sub(r"[^a-zA-Z\s\-]", "", text) # Remove punctuation and numbers (but keep -)
        words = [self.stemmer.stem(w.lower()) # Do stemming
                 for w in nltk.word_tokenize(raw_text) # Word tokenize 
                 if w not in self.stop_words] # Only keep words not on stopwords list 
        return words
        
        
    # Gets the features from the training documents and stores them 
    # Features are the most common words 
    def set_features(self): 
        training_documents = data_setup_and_fetch.fetch_part1_document_training()        
        all_words = []
        # Get all words from training documents, and use most common 1000 as features 
        for doc in training_documents:
            words = self.get_tokens(doc[1])
            all_words += words            
        freq_dist = nltk.FreqDist(all_words)         
        common_words = freq_dist.most_common(500) 
        for (word, count) in common_words: 
            self.word_features.add(word)
    
    
    # Input: String text, String label 
    # Output: ({String, Bool}, String)
    # Returns a tuple of the form ({word: boolean}, label) 
    #   words are features,values are true or false based on if word in text
    def get_labeled_feature_set(self, text, label):
        words = self.get_tokens(text) 
        feature_set = {} 
        for w in self.word_features: 
            if w in words:
                feature_set[w] = True 
            else: 
                feature_set[w] = False                 
        labeled_feature_set = (feature_set, label)
        return labeled_feature_set


    # Given text training_data, returns list of tuples (the training set)
    def get_feature_sets(self, training_data): 
        feature_sets = []
        # Get feature sets
        for training_element in training_data:
            label = training_element[0].split("_")[2].split(".")[0]
            feature_sets.append(self.get_labeled_feature_set(training_element[1], label))
        return feature_sets
    
    
    # Returns the avg. error through 10-fold cross validation for each model 
    # training_data can be for documents, paragraphs, or sentences 
    def k_fold(self, training_data): 
        training_set = self.get_feature_sets(training_data)
        random.shuffle(training_set)
        nb_error = 0
        dt_error = 0
        maxent_error = 0
        index = 0
        fold_length = len(training_set)//self.k # Takes 10 folds, rounds down length (because it may not be possible to get folds of same length)
        for k in range(self.k): 
            training_folds = []
            testing_fold = []
            if k == self.k-1: 
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
            dt_classifier = nltk.DecisionTreeClassifier.train(training_folds)
            maxent_classifier = nltk.MaxentClassifier.train(training_folds, max_iter=10)     
            nb_error += self.get_error(testing_fold, nb_classifier)   
            dt_error += self.get_error(testing_fold, dt_classifier)
            maxent_error += self.get_error(testing_fold, maxent_classifier)
        # Average the errors and display
        nb_error = nb_error / self.k
        dt_error = dt_error / self.k
        maxent_error = maxent_error / self.k 
        print("Naive Bayes Avg. Error: " + str(nb_error))
        print("Decision Tree Avg. Error: " + str(dt_error))
        print("Maxent Avg. Error: " + str(maxent_error))     
        min_error = min(nb_error, min(dt_error, maxent_error))
        print("Min Error: " + str(min_error))
    
    
    # Returns the error of classifier on test_set 
    # The error is just the number of incorrect classifications 
    def get_error(self, test_set, classifier):
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


    # Displays error for each model (using 10-fold cross validation)
    def get_best_models(self): 
        training_documents = data_setup_and_fetch.fetch_part1_document_training()
        training_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_sentences = data_setup_and_fetch.fetch_part1_paragraph_training()
        print("DOCUMENTS: ")
        self.k_fold(training_documents)
        print("PARAGRAPHS: ")
        self.k_fold(training_paragraphs)
        print("SENTENCES: ")
        self.k_fold(training_sentences)
        

    # Trains with documents (using predetermined optimal model: Naive Bayes)
    def train_with_documents(self):
        training_documents = data_setup_and_fetch.fetch_part1_document_training()
        training_set = self.get_feature_sets(training_documents)
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        self.document_classifier = nb_classifier
        
        
    # Trains with paragraphs (using predetermined optimal model: Naive Bayes)
    def train_with_paragraphs(self): 
        training_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = self.get_feature_sets(training_paragraphs)
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        self.paragraph_classifier = nb_classifier 
                
                
    # Trains with sentences (using predetermined optimal model: Naive Bayes)
    def train_with_sentences(self): 
        training_sentences = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = self.get_feature_sets(training_sentences)
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        self.sentence_classifier = nb_classifier           
        
        
    # Tests document classifier 
    def test_with_documents(self): 
        testing_documents = data_setup_and_fetch.fetch_part1_document_testing()
        testing_set = self.get_feature_sets(testing_documents)
        print("Naive Bayes, Accuracy: " + str(nltk.classify.accuracy(self.document_classifier, testing_set)))
        self.display_stats(testing_set, self.document_classifier)
            
            
    # Tests paragraph classifier
    def test_with_paragraphs(self): 
        testing_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_testing()
        testing_set = self.get_feature_sets(testing_paragraphs)
        print("Naive Bayes, Accuracy: " + str(nltk.classify.accuracy(self.paragraph_classifier, testing_set)))
        self.display_stats(testing_set, self.paragraph_classifier)
    
    
    # Tests setence classifier
    def test_with_sentences(self): 
        testing_sentences = data_setup_and_fetch.fetch_part1_sentence_testing()
        testing_set = self.get_feature_sets(testing_sentences)
        print("Naive Bayes, Accuracy: " + str(nltk.classify.accuracy(self.sentence_classifier, testing_set)))
        self.display_stats(testing_set, self.sentence_classifier)
            
    
    # Displays accuracy, precision, recall, and F-measure for a classifier given a test_set
    def display_stats(self, test_set, classifier):
        golden_labels = []
        predicted_labels = []
        for data_element in test_set:
            golden_labels.append(data_element[1]) # label 
            predicted_labels.append(classifier.classify(data_element[0])) # predict label for element
        confusion_matrix = nltk.ConfusionMatrix(golden_labels, predicted_labels)
        print(confusion_matrix.evaluate())
        
        
if __name__ == "__main__": 
    topic_classifier = TopicClassifier()
    #topic_classifier.get_best_models()
    topic_classifier.train_with_documents()
    topic_classifier.train_with_paragraphs()
    topic_classifier.train_with_sentences()
    print("________________________________________________________________")
    print("DOCUMENTS: ")
    topic_classifier.test_with_documents()
    print("________________________________________________________________")
    print("\nPARAGRAPHS: ")
    topic_classifier.test_with_paragraphs()
    print("________________________________________________________________")
    print("\nSENTENCES: ")
    topic_classifier.test_with_sentences()
    print("________________________________________________________________")