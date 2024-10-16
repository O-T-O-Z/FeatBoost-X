import time

from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from featboostx import FeatBoostRegressor

start_time = time.time()
X, y = make_regression(
    n_samples=10000,
    n_features=1000,
    n_informative=2,
    random_state=0,
    shuffle=True,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = FeatBoostRegressor(
    XGBRegressor(),
    metric="mae",
    verbose=0,
    siso_ranking_size=10,
    max_number_of_features=2,
)
clf.fit(X_train, y_train)
print(clf.selected_subset_)

clf1 = XGBRegressor()
clf1.fit(X_train[:, clf.selected_subset_], y_train)
print(mean_absolute_error(y_test, clf1.predict(X_test[:, clf.selected_subset_])))

clf2 = XGBRegressor()
clf2.fit(X_train, y_train)
print(mean_absolute_error(y_test, clf2.predict(X_test)))

print("--- %s seconds ---" % (time.time() - start_time))
