import numpy as np
import random
from nltk import word_tokenize, WordNetLemmatizer
from collections import Counter
from nltk import NaiveBayesClassifier, classify

#Loading in stopword file (created from suggested website with added punct.)
sw = "/Users/Jessica/Documents/Python/stopwords.txt"
stops = np.loadtxt(sw, delimiter='\n', dtype='str')
stoplist = []
for row in stops:
    stoplist.append(row)

#Preprocessing function - using tokenizers and lemmatizers
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]
    

#Preprocesses text, does not count stop words, gets count. Bag of words.
def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
         

#Taking set of features and proportion of examples
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    dev_size = int(len(features) * ((1 - samples_proportion)/2) + train_size)
    train_set, dev_set, test_set = features[:train_size], features[train_size:dev_size], features[dev_size:]
    print ('Training set size = ' + str(len(train_set)) + ' reviews')
    print ('Development set size = ' + str(len(dev_set)) + ' reviews')
    print ('Test set size = ' + str(len(test_set)) + ' reviews')
    #train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, dev_set, test_set, classifier
    

#Evaluate development set on train set
def evaluate(train_set, dev_set, classifier):
    # check how the classifier performs on the training and test sets
    print ('Accuracy on the train set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the development set = ' + str(classify.accuracy(classifier, dev_set)))
    classifier.show_most_informative_features(20)
    

#Test test set on train set
def test(train_set, test_set, classifier):
    # check how the classifier performs on the test set
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    classifier.show_most_informative_features(20)  


#Setting up main executable method
def main():
    #Reading in pos/neg files
    pos = "/Users/Jessica/Documents/Python/rt-polarity.pos.txt"
    neg = "/Users/Jessica/Documents/Python/rt-polarity.neg.txt"
    #Separating by line
    posclass = np.loadtxt(pos, delimiter='\n', dtype='str')
    negclass = np.loadtxt(neg, delimiter='\n', dtype='str')
    posCorpus = []
    negCorpus = []
    #Putting each line into array
    for row in posclass:
        temp = row.split('\n')
        posCorpus.extend(temp)
    
    for row in negclass:
        temp = row.split('\n')
        negCorpus.extend(temp)
 
    #Combining the two arrays, adding class labels, shuffling
    all_reviews = [(review, 'pos') for review in posCorpus]
    all_reviews += [(review, 'neg') for review in negCorpus]
    random.shuffle(all_reviews)
    
    print('Corpus size = ' + str(len(all_reviews)) + ' reviews')
         
    #Extract features from reviews and pairs with class label
    all_features = [(get_features(review, 'bow'), label) for (review, label) in all_reviews]
    
    #Train classifier
    train_set, dev_set, test_set, classifier = train(all_features, 0.7)
    
    #Evaluate performance on development set
    #evaluate(train_set, dev_set, classifier)
    
    #Evaluate performance on test set
    test(train_set, test_set, classifier)
 
if __name__ == "__main__":
    main()


























