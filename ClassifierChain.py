import pandas as pd
from sklearn.multioutput import ClassifierChain 
from sklearn.datasets import make_multilabel_classification as make_ml_clf
from sklearn.linear_model import LogisticRegression, MultiTaskLassoCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score,plot_roc_curve
from sklearn.preprocessing import scale
from sklearn.multioutput import ClassifierChain


df = pd.read_csv('simulated_df.csv')
predictors_simulated = df.drop(['FIBR_PREDS', 'ZSN', 'LET_IS'], axis = 1)
targets_simulated = df[['FIBR_PREDS', 'ZSN', 'LET_IS']]

# Preparing & Scaling predictors for the regression
X = scale(predictors_simulated.values)

# Preparing the targets
y = targets_simulated.values


# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=15)

# Thaining & fitting Multi-task Logit model on the original dataset 
base_lr = LogisticRegression(penalty= 'l2',solver='lbfgs', random_state=15)
chain = ClassifierChain(base_lr, order=[0,1,2], random_state=15, cv = 10)
mtl_log_model = chain.fit(X_train, y_train)
pred_mtl_log= mtl_log_model.predict(X_test)
