import time

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from featboostx import FeatBoostClassifier

start_time = time.time()
X, y = make_classification(
    n_samples=1000,
    n_features=10000,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=True,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = FeatBoostClassifier(
    XGBClassifier(),
    loss="softmax",
    verbose=2,
    siso_ranking_size=100,
    max_number_of_features=10,
    num_resets=1,
    # use_shap=True,
)
clf.fit(X_train, y_train)
print(clf.selected_subset_)

clf1 = XGBClassifier()
clf1.fit(X_train[:, clf.selected_subset_], y_train)
print(classification_report(y_test, clf1.predict(X_test[:, clf.selected_subset_])))

clf2 = XGBClassifier()
clf2.fit(X_train, y_train)
print(classification_report(y_test, clf2.predict(X_test)))

print("--- %s seconds ---" % (time.time() - start_time))
