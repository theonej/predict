import nltk
import argparse
import data.train_data as train_data

DATA_SIZE = 2000
FEATURE_COUNT = 2000

def get_features(data):
	all_contents = [item['contents'].split() for item in data]

	all_words = [word.lower() for wordlist in all_contents for word in wordlist]

	frequency = nltk.FreqDist(all_words)

	return list(set(list(frequency)))[:FEATURE_COUNT]

def get_content_features(contents, features):
	contents_words = set(contents)

	contents_features = {};
	for feature in features:
		contents_features['contains({})'.format(feature)] = (feature in contents_words)

	return contents_features
'''
def label_with_class(item, features):
	item_contents = get_content_features(item['contents'], features)
	class_name = item['className']
	return  (
				dict(contents=item_contents),
				class_name
			)
'''
def label_with_class(item, features):
	item_contents = get_content_features(item['contents'], features)
	class_name = item['className']
	return  (
				item_contents,
				class_name
			)
'''
def label_with_sub_class(item, features):
	item_contents = get_content_features(item['contents'], features)
	sub_class = item['subClass']
	return  (
				dict(contents=item_contents),
				sub_class
			)
'''

def label_with_sub_class(item, features):
	item_contents = get_content_features(item['contents'], features)
	sub_class = item['subClass']
	return  (
				dict(contents=item_contents),
				sub_class
			)


def train_section_class_classififer(section):
	train, test = train_data.get_section_data(section, 30)

	features = get_features(train)

	class_train_data = [label_with_class(item, features) for item in train[:DATA_SIZE]]
	class_test_data = [label_with_class(item, features) for item in test[:DATA_SIZE]]

	classifier = nltk.classify.NaiveBayesClassifier.train(class_train_data)

	prediction = classifier.classify(class_test_data[0][0])
	truth = class_test_data[0][1]
	print("predicted class: %s.  Actual class: %s"%(prediction, truth))

	accuracy = nltk.classify.accuracy(classifier, class_test_data)
	print("Class model accuracy: %s"%accuracy)

def train_section_subclass_classififer(section):
	train, test = train_data.get_section_data(section, 30)

	sub_class_train_data = [label_with_sub_class(item) for item in train]
	sub_class_test_data = [label_with_sub_class(item) for item in test]

	classifier = nltk.classify.NaiveBayesClassifier.train(sub_class_train_data)

	prediction = classifier.classify(sub_class_train_data[0][0])
	truth = sub_class_test_data[0][1]
	print("predicted sub class: %s.  Actual sub class: %s"%(prediction, truth))

	accuracy = nltk.classify.accuracy(classifier, sub_class_test_data)
	print("Subclass model accuracy: %s"%accuracy)

parser = argparse.ArgumentParser()
parser.add_argument("section", help="The section for which you want to train a model")
section = parser.parse_args().section


train_section_class_classififer(section)
#train_section_subclass_classififer(section)



