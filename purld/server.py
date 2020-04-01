# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, median_absolute_error, \
    mean_absolute_error, \
    max_error, \
    balanced_accuracy_score, \
    precision_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
url = "./dataset.csv"
names = ['having_IPhaving_IP_Address',
         'URLURL_Length',
         'Shortining_Service',
         'having_At_Symbol',
         'double_slash_redirecting',
         'Prefix_Suffix',
         'having_Sub_Domain',
         'SSLfinal_State',
         'Domain_registeration_length',
         'Favicon',
         'port',
         'HTTPS_token',
         'Request_URL',
         'URL_of_Anchor',
         'Links_in_tags',
         'SFH',
         'Submitting_to_email',
         'Abnormal_URL',
         'Redirect',
         'on_mouseover',
         'RightClick',
         'popUpWidnow',
         'Iframe',
         'age_of_domain',
         'DNSRecord',
         'web_traffic',
         'Page_Rank',
         'Google_Index',
         'Links_pointing_to_page',
         'Statistical_report',
         'Result']
dataset = read_csv(url, names=names)

# shape
# print('Dataset shape :')
# print(dataset.shape)

# head
# print('Dataset head(30) :')
# print(dataset.head(40))

# descriptions
# print('Dataset describtions :')
# print(dataset.describe())

# class distribution
print("Dataset groupby('Result') :")
print(dataset.groupby('Result').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# pyplot.show()

# histograms
# dataset.hist()
# pyplot.show()

# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20,
                                                                random_state=1)

# print(X)
# print(y)
# print()
#
# print(X_train)
# print(Y_train)
#
# print(X_validation)
# print(Y_validation)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
print('evaluate each model in turn :')
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print('predictions :')
print(predictions)

# Evaluate predictions
print('\nAccuracy Score :')
print(accuracy_score(Y_validation, predictions))

print('\nConfusion Matrix :')
print(confusion_matrix(Y_validation, predictions))

print('\nClassification Report :')
print(classification_report(Y_validation, predictions))

print('\nMean Squared Error :')
print(mean_squared_error(Y_validation, predictions))

print('\nMean Absolute Error :')
print(mean_absolute_error(Y_validation, predictions))

print('\nMedian Absolute Error :')
print(median_absolute_error(Y_validation, predictions))

print('\nMax Error :')
print(max_error(Y_validation, predictions))

print('\nBalanced Accuracy Score :')
print(f"{balanced_accuracy_score(Y_validation, predictions)}")

print('\nPrecision Score (average = marcro) :')
print(precision_score(Y_validation, predictions, average='macro'))

print('\nPrecision Score (average = micro) :')
print(precision_score(Y_validation, predictions, average='micro'))

print('\nPrecision Score (average = weighted) :')
print(precision_score(Y_validation, predictions, average='weighted'))

print('\nPrecision Score (average = None) :')
print(precision_score(Y_validation, predictions, average=None))

print('\nr2 Score :')
print(r2_score(Y_validation, predictions))

voting_clf = VotingClassifier(estimators=models, voting='hard')
voting_clf.fit(X_train, Y_train)
y_pred = voting_clf.predict(X_validation)
print(f"\nVoting Classifier's accuracy: {accuracy_score(y_pred, Y_validation)}")

# print('\nVotingClassifier :')
# print(VotingClassifier(estimators=[Y_validation, predictions], voting='hard'))

# clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()
#
# eclf1 = VotingClassifier(estimators=[
#     ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

# eclf1 = VotingClassifier(estimators=[Y_validation, predictions], voting='hard');
# eclf1 = eclf1.fit(X, y)
# print(eclf1.predict(X))
# print(X, y)
