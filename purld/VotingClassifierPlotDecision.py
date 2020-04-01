from itertools import product
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

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

array = dataset.values
X = array[:, [0, 2]]
y = array[:, 4]
# X = dataset.values[:, 0:4]
# y = dataset.target
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20,
#                                                                 random_state=1)

# Training classifiers
# clf1 = DecisionTreeClassifier(max_depth=4)
# clf2 = KNeighborsClassifier(n_neighbors=7)
# clf3 = SVC(gamma=.1, kernel='rbf', probability=True)

clf1 = LogisticRegression(max_iter=1000, random_state=123)
clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                    ('gnb', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

# for idx, clf, tt in zip(product([0, 1], [0, 1]),
#                         [clf1, clf2, clf3, eclf],
#                         ['Decision Tree (depth=4)', 'KNN (k=7)',
#                          'Kernel SVM', 'Hard Voting']):
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['LogisticRegression', 'RandomForestClassifier',
                         'GaussianNB', 'Hard Voting']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
