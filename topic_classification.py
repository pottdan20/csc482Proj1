import nltk
import data_setup_and_fetch

class TopicClassifier:
    def __init__(self):
        self.document_classifiers = {}
        self.paragraph_classifiers = {}
        self.sentence_classifiers = {}
    
    # Input: String text, String label 
    # Output: List[(List[String], String)]
    # Returns a list of tuples, where each tuple is ({features}, label)
    def get_labeled_feature_set(self, text, label):
        words = nltk.word_tokenize(text)
        feature_set = {w : w for w in set(words)}
        labeled_feature_set = (feature_set, label)
        return labeled_feature_set

    # Trains multiple models with documents 
    def train_with_documents(self):
        training_documents = data_setup_and_fetch.fetch_part1_document_training()
        training_set = []
        # Get feature set for each document
        for doc in training_documents:
            label = doc[0].split("_")[2].split(".")[0]
            training_set.append(self.get_labeled_feature_set(doc[1], label))
        # Train classifiers 
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.document_classifiers["Naive Bayes"] = nb_classifier
        self.document_classifiers["Decision Tree"] = dt_classifier
        self.document_classifiers["Max Entropy"] = maxent_classifier

    # Trains multiple models with paragraphs 
    def train_with_paragraphs(self): 
        training_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = []
        # Get feature set for each paragraph 
        for par in training_paragraphs: 
            label = par[0].split("_")[2].split(".")[0]
            training_set.append(self.get_labeled_feature_set(par[1], label))
        # Train classifiers
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.paragraph_classifiers["Naive Bayes"] = nb_classifier
        self.paragraph_classifiers["Decision Tree"] = dt_classifier
        self.paragraph_classifiers["Max Entropy"] = maxent_classifier
                
    # Trains multiple models with sentences
    def train_with_sentences(self): 
        training_sentences = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = []
        # Get feature set for each sentence
        for sent in training_sentences: 
            label = sent[0].split("_")[2].split(".")[0]
            training_set.append(self.get_labeled_feature_set(sent[1], label))
        # Train classifiers 
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.sentence_classifiers["Naive Bayes"] = nb_classifier
        self.sentence_classifiers["Decision Tree"] = dt_classifier
        self.sentence_classifiers["Max Entropy"] = maxent_classifier
        
    def get_best_model_documents(self): 
        pass
    
    def get_best_model_paragraphs(self): 
        pass 
    
    def get_best_model_sentences(self): 
        pass
        
    # Tests document classifiers 
    def test_with_documents(self): 
        testing_documents = data_setup_and_fetch.fetch_part1_document_testing()
        testing_set = []
        # Get feature sets for each test document
        for doc in testing_documents:
            label = doc[0].split("_")[2].split(".")[0]
            testing_set.append(self.get_labeled_feature_set(doc[1], label))
        for classifier_name in self.document_classifiers:
            classifier = self.document_classifiers[classifier_name]
            print(classifier_name + ", Accuracy: " + str(nltk.classify.accuracy(classifier, testing_set)))
            self.display_stats(testing_set, classifier)
            
    # Tests paragraph classifiers 
    def test_with_paragraphs(self): 
        testing_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_testing()
        testing_set = []
        # Get feature sets for each test paragraph
        for par in testing_paragraphs:
            label = par[0].split("_")[2].split(".")[0]
            testing_set.append(self.get_labeled_feature_set(par[1], label))
        for classifier_name in self.paragraph_classifiers:
            classifier = self.paragraph_classifiers[classifier_name]
            print(classifier_name + ", Accuracy: " + str(nltk.classify.accuracy(classifier, testing_set)))
            self.display_stats(testing_set, classifier)
    
    # Tests setence classifiers 
    def test_with_sentences(self): 
        testing_sentences = data_setup_and_fetch.fetch_part1_sentence_testing()
        testing_set = []
        # Get feature sets for each test sentence 
        for sent in testing_sentences:
            label = sent[0].split("_")[2].split(".")[0]
            testing_set.append(self.get_labeled_feature_set(sent[1], label))
        for classifier_name in self.sentence_classifiers:
            classifier = self.paragraph_classifiers[classifier_name]            
            print(classifier_name + ", Accuracy: " + str(nltk.classify.accuracy(classifier, testing_set)))
            self.display_stats(testing_set, classifier)
            
    def display_stats(self, test_set, classifier):
        golden_labels = []
        predicted_labels = []
        for data_element in test_set:
            golden_labels.append(data_element[1]) # label 
            predicted_labels.append(classifier.classify(data_element[0])) # predict label for element
        confusion_matrix = nltk.ConfusionMatrix(golden_labels, predicted_labels)
        print(confusion_matrix.evaluate())
    
    def save_classifiers(self): 
        # to save classifier     
        # import pickle
        # f = open('my_classifier.pickle', 'wb')
        # pickle.dump(classifier, f)
        # f.close()
        pass
                                    
    
    def restore_classifiers(self): 
        # to restore classifier 
        #     import pickle
        # f = open('my_classifier.pickle', 'rb')
        # classifier = pickle.load(f)
        # f.close()
        pass

if __name__ == "__main__": 
    topic_classifier = TopicClassifier()
    topic_classifier.train_with_documents()
    topic_classifier.test_with_documents()
    topic_classifier.train_with_paragraphs()
    topic_classifier.test_with_paragraphs()
    topic_classifier.train_with_sentences()
    topic_classifier.test_with_sentences()

    # DO CROSS VALIDATION 

        
        
