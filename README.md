# FeatBoost-X
Python implementation of FeatBoost-X. See the [paper]() for details.


## Usage
```shell
pip install featboostx
```

### Example
```python
from featboostx import FeatBoostClassifier

clf = FeatBoostClassifier()
clf.fit(X, y)
print(clf.selected_subset_)
```
For a more detailed example, see the [classification example](examples/example_classification.py) or the 
[regression example](examples/example_regression.py).

## Feature selection methods
FeatBoost-X is available classification, regression, and survival problems.
- Classification supports the objectives accuracy (`acc`) and the F1-score (`f1`) through the `FeatBoostClassifier`-class. These can be optimized through the `softmax` or `adaboost` objective.
This implementation originates from the Python implementation of the [original paper](https://github.com/amjams/FeatBoost).
- Regression supports the `mae` objective through the `FeatBoostRegressor`-class and can be optimized through `adaptive` boosting.
- Survival supports the `c_index` objective through the `FeatBoostRegressor`-class and can be optimized through `adaptive` boosting.


# Illustration of FeatBoost-X
![Figure 1](images/Figure_1.png)