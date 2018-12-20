from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


import data


def parse_dataset(dataset, word_num):
    input_x, output_y = [], []
    for _, indexes, _, r in dataset:
        x = [0] * word_num
        for idx in indexes:
            x[idx] = 1
        input_x.append(x)
        output_y.append(r)
    return input_x, output_y


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(criterion='entropy'),
    RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

word_corpus = data.WordCorpus('data/negotiate/')
train_dataset = word_corpus.train
val_dataset = word_corpus.valid
test_dataset = word_corpus.test

total_word_num = len(word_corpus.word_dict)

X_train, y_train = parse_dataset(train_dataset, total_word_num)
X_valid, y_valid = parse_dataset(val_dataset, total_word_num)
X_test, y_test = parse_dataset(test_dataset, total_word_num)


# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name)
    print(clf)
    print(score)
