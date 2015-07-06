import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve


# Step 1: Load data, look around ##################################################


messages = [line.rstrip() for line in open('SMSSpamCollection')]
# print(len(messages))

# for message_no, message in enumerate(messages[:10]):
#     print(message_no, message)

messages = pandas.read_csv('SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
# print(messages)
# print(messages.groupby('label').describe())

messages['length'] = messages['message'].map(lambda text: len(text))
# print(messages.head())
messages.length.hist(bins=20)
# plt.show()
# print(messages.length.describe())

# print(list(messages.message[messages.length > 900]))
messages.hist(column='length', by='label', bins=50)
# plt.show()

# Step 2: Data pre-processing #####################################################
# https://en.wikipedia.org/wiki/Bag-of-words_model
# http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
# https://en.wikipedia.org/wiki/Lemmatisation


def split_into_tokens(message):
    return TextBlob(message).words

# print(messages.message.head())
# print(messages.message.head().apply(split_into_tokens))

# print(TextBlob("Hello world, how is it going?").tags)


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]
# print(messages.message.head().apply(split_into_lemmas))


# Step 3: Data to vectors ######################################################
# In the bag-of-words model:
# counting how many times does a word occur in each message (term frequency)
# weighting the counts, so that frequent tokens get lower weight (inverse document frequency)
# normalizing the vectors to unit length, to abstract from the original text length (L2 norm)
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
# print(len(bow_transformer.vocabulary_))

message4 = messages['message'][3]
# print(message4)
bow4 = bow_transformer.transform([message4])
# print(bow4)
# print(bow_transformer.get_feature_names()[6736])
# print(bow_transformer.get_feature_names()[8013])

messages_bow = bow_transformer.transform(messages['message'])
# print('sparse matrix shape:', messages_bow.shape)
# print('number of non-zeros:', messages_bow.nnz)
# print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
# print(tfidf4)
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)
# print(messages_tfidf.shape)


#  Step 4: Training a model, detecting spam #######################################################
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://en.wikipedia.org/wiki/Precision_and_recall
# https://en.wikipedia.org/wiki/F1_score

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
# print('predicted:', spam_detector.predict(tfidf4)[0])
# print('expected:', messages.label[3])

all_predictions = spam_detector.predict(messages_tfidf)
# print(all_predictions)

# print('accuracy', accuracy_score(messages['label'], all_predictions))
# print('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
# print('(row=expected, col=predicted)')

# plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
# plt.title('confusion matrix')
# plt.colorbar()
# plt.ylabel('expected label')
# plt.xlabel('predicted label')
# plt.show()

# print(classification_report(messages['label'], all_predictions))


# Step 5: How to run experiments? ################################################

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

# print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


def main():
    scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                             msg_train,  # training data
                             label_train,  # training labels
                             cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring='accuracy',  # which scoring metric?
                             n_jobs=-1,  # -1 = use all cores = faster
                             )
    print(scores)
    print(scores.mean(), scores.std())

if __name__ == '__main__':
    main()
