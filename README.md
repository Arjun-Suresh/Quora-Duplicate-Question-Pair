# Quora-Duplicate-Question-Pair
* The application is written in Python 3.4.
* The code includes data cleanup, feature extraction and classification using Naive Bayes, SVM and XGBoost algorithms.
* It uses the following external libraries:
  * nltk
  * pandas
  * sklearn
  * numpy
  * nltk.corpus which has the stopwords
  * nltk.tag.stanford
  * xgboost
  * matplotlib
* The NER tagger uses a corpus which is present as a .jar file (handled within the code). To use this, the system needs to have java installed and JAVA_HOME environment variable referring to the location of the jdk folder.
* The “stanford-ner-2014-06-16” folder contains these .jar files and needs to be present in the same folder as the python files.
* The dataset is present in the input folder as train.csv file.
* The application can be run as “python Classifier.py Input\train.csv 1”.
  * The first parameter is the location of the dataset.
  * The second parameter refers to the model: 1 for naïve bayes, 2 for XGBoost and 3 for svm.
