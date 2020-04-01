# Load libraries
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
url = "./dataset.csv"
dataset = read_csv(url)

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20,
                                                                random_state=1)

# Spot Check Algorithms
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('RFC', RandomForestClassifier()),
    ('Linear SVM', SVC(kernel='linear')),
    ('AdaBoost', AdaBoostClassifier()),
    ('Neural Net', MLPClassifier()),
    ('Gaussian Process', GaussianProcessClassifier())
]

# Evaluate Each Algorithms Accuracy :
for model_tuple in models:
    model = model_tuple[1]
    if 'random_state' in model.get_params().keys():
        model.set_params(random_state=1)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_validation)
    acc = accuracy_score(y_pred, Y_validation)
    print(f"{model_tuple[0]} : {acc}")

# Evaluate Each Algorithms cv mean & cv std Results :
print('Evaluate Each Algorithms cv mean & cv std Results :')
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: (cv mean results %f) (cv std results %f)' % (name, cv_results.mean(),
                                                            cv_results.std()))

# Evaluate Pair Voting Accuracy
voting_clf = VotingClassifier(estimators=models, voting='hard')
voting_clf.fit(X_train, Y_train)
y_pred = voting_clf.predict(X_validation)
print(f"\nVoting Classifier's accuracy: {accuracy_score(y_pred, Y_validation)}")
