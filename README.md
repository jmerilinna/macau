# MACAU
The MACAU [1] Python package takes pre-fitted LightGBM classifiers/regressors as input and extends its capabilities during inference. Beyond providing probabilities or regressed values, MACAU offers the following outputs:

- Prediction: The predicted probability or regressed value.
- Inference Novelty: Indicates how unusual the predicted value is, expressed as a z-score.
- Novelty: Measures the uniqueness of the features in the predicted sample within its local context, represented as a probability being unique.
- Conditional Novelty: Assesses the oddness of the predicted sample's features in its local context, considering only the used features and expressed as probability.
- Inference Novelty uncertainty: Represents the standard deviation of the inference novelty.
- Novelty uncertainty: Reflects the standard deviation of novelty.
- Conditional Novelty uncertainty: Signifies the standard deviation of conditional novelty.
- Aleatoric Uncertainty: Provides 1-sigma confidence intervals based on the aleatoric uncertainty of the predicted sample.
- Epistemic Uncertainty: Supplies 1-sigma confidence intervals based on the epistemic uncertainty of the predicted sample.
- Uncertainty: Encompasses both aleatoric and epistemic uncertainty, providing a comprehensive measure.

[1] https://aircconline.com/csit/papers/vol13/csit131920.pdf

## Install
<pre>
git clone https://github.com/jmerilinna/macau.git
pip install ./macau
</pre>
## Usage

### Regression
<pre>
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from macau import MACAU
import lightgbm

# Create synthetic dataset
X, Y = make_moons(n_samples=1000, noise=0.1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Initialize LightGBM model
model = lightgbm.LGBMRegressor(boosting_type='rf',
                                importance_type='gain',
                                n_estimators=100,
                                max_depth=-1,
                                colsample_bytree=1.0,
                                subsample_freq=1,
                                subsample=0.8,
                                reg_alpha=0,
                                reg_lambda=0,
                                verbose=-1,
                                num_leaves=31,
                                min_child_samples=20,
                                n_jobs=10)

# Fit LightGBM model
model.fit(X_train, Y_train)

# Initialize and fit MACAU model
macau = MACAU(model).fit(X_train, Y_train)

# Make predictions using MACAU
result = macau.predict(X_test)
</pre>


### Classification
<pre>
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from macau import MACAU
import lightgbm

# Create synthetic dataset
X, Y = make_moons(n_samples=1000, noise=0.1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Initialize LightGBM model
model = lightgbm.LGBMClassifier(boosting_type='rf',
                                importance_type='gain',
                                n_estimators=100,
                                max_depth=-1,
                                colsample_bytree=1.0,
                                subsample_freq=1,
                                subsample=0.8,
                                reg_alpha=0,
                                reg_lambda=0,
                                verbose=-1,
                                num_leaves=31,
                                min_child_samples=20,
                                n_jobs=10)

# Fit LightGBM model
model.fit(X_train, Y_train)

# Initialize and fit MACAU model
macau = MACAU(model).fit(X_train, Y_train)

# Make predictions using MACAU
result = macau.predict(X_test)
</pre>
