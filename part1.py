import nltk
import data_setup_and_fetch

class TopicClassifier: 
    def __init__(self):         
        self.document_classifiers = []
        self.paragraph_classifiers = []
        self.sentence_classifiers = []
    
    # Input: String text, String label 
    # Output: List[(List[String], String)]
    # Returns a list of tuples, where each tuple is ([features], label)
    def extract_features(self, text, label):
        words = nltk.word_tokenize(text)
        features = [w for w in set(words)]        
        labeled_features = (features, label)
        return labeled_features

    def train_with_documents(self):
        training_documents = data_setup_and_fetch.fetch_part1_document_training()
        training_set = []
        for doc in training_documents:
            label = doc[0].split("_")[2]
            training_set.append(self.extract_features(doc[1], label))
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.document_classifiers.append(nb_classifier)
        self.document_classifiers.append(dt_classifier)
        self.document_classifiers.append(maxent_classifier)
        
    def train_with_paragraphs(self): 
        training_paragraphs = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = []
        for par in training_paragraphs: 
            label = par[0].split("_")[2]
            training_set.append(self.extract_features(par[1], label))
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.paragraph_classifiers.append(nb_classifier)
        self.paragraph_classifiers.append(dt_classifier)
        self.paragraph_classifiers.append(maxent_classifier)
        
    def train_with_sentences(self): 
        training_sentences = data_setup_and_fetch.fetch_part1_paragraph_training()
        training_set = []
        for sent in training_sentences: 
            label = sent[0].split("_")[2]
            training_set.append(self.extract_features(sent[1], label))
        nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
        dt_classifier = nltk.DecisionTreeClassifier.train(training_set)
        maxent_classifier = nltk.MaxentClassifier.train(training_set)
        self.sentence_classifiers.append(nb_classifier)
        self.sentence_classifiers.append(dt_classifier)
        self.sentence_classifiers.append(maxent_classifier)
        
    def test_with_documents(self): 
        testing_documents = data_setup_and_fetch.fetch_part1_document_training()
        testing_set = []
        for doc in testing_documents:
            label = doc[0].split("_")[2]
            testing_set.append(self.extract_features(doc[1], label))
        for classifier in self.document_classifiers:
            print('Accuracy: ' + str(nltk.classify.accuracy(classifier, testing_set)))
            
    def test_with_paragraphs(self): 
        pass
    
    def test_with_sentences(self): 
        pass
    
    
    
# >>> classifier = nltk.NaiveBayesClassifier.train(train_set) 
# >>> print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 
# 0.75
 	
# >>> def tag_list(tagged_sents):
# ...     return [tag for sent in tagged_sents for (word, tag) in sent]
# >>> def apply_tagger(tagger, corpus):
# ...     return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
# >>> gold = tag_list(brown.tagged_sents(categories='editorial'))
# >>> test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
# >>> cm = nltk.ConfusionMatrix(gold, test)
# >>> print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

if __name__ == "__main__": 
    nltk.download()
    # topic_classifier = TopicClassifier()
    # topic_classifier.train_with_documents()
    # topic_classifier.test_with_documents()

        
        
