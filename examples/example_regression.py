import time

from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from statsmodels.tools.tools import add_constant

from featboostx import FeatBoostRegressor

start_time = time.time()
X, y = make_regression(
    n_samples=100,
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
    verbose=2,
    siso_ranking_size=20,
    max_number_of_features=2,
    num_resets=1,
    use_shap=True,
    corr_check=True,
)
clf.fit(X_train, y_train)
print(clf.selected_subset_)

clf1 = XGBRegressor()
clf1.fit(X_train[:, clf.selected_subset_], y_train)
print(mean_absolute_error(y_test, clf1.predict(X_test[:, clf.selected_subset_])))
data = pd.DataFrame(X_train[:, clf.selected_subset_])
variable = add_constant(data)
vif_data = pd.DataFrame(
    {
        "Variable": variable.columns,
        "VIF": [
            variance_inflation_factor(variable.values, i)
            for i in range(variable.shape[1])
        ],
    }
)
print("VIF for the selected variables:")
print(vif_data)

clf2 = XGBRegressor()
clf2.fit(X_train, y_train)
print(mean_absolute_error(y_test, clf2.predict(X_test)))

print("--- %s seconds ---" % (time.time() - start_time))
