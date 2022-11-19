import numpy as np
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, plot_roc_curve, recall_score, precision_score
from scipy import stats

df_simulated = pd.read_csv('simulated_df')

y = df_simulated[['FIBR_PREDS', 'ZSN', 'LET_IS']]
x = df_simulated[['NITR_S', 'K_SH_POST', 'zab_leg_01', 'ZSN_A', 'n_r_ecg_p_05', 'AGE', 'SEX']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

nparam = 7
nlabel = 3

with pm.Model() as logreg_model:
    X = pm.Data('X', x_train)
    Y = pm.Data('Y', y_train)

    σ = pm.HalfCauchy.dist(beta=2.5, shape=nlabel)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=1, n=nlabel, sd_dist=σ)
    chol = pm.expand_packed_triangular(nlabel, packed_chol, lower=True)
    Σ = pm.Deterministic('cov', tt.dot(chol, chol.T))

    α = pm.Cauchy('intercept', alpha=0, beta=10, shape=nlabel)
    mu = np.zeros(nlabel)
    β = pm.MvNormal('coef', mu=mu, cov=Σ, shape=(nparam, nlabel))

    θ = pm.invlogit(X.dot(β) + α)
    ŷ = pm.Bernoulli('y_obs', p=θ, observed=Y)

with logreg_model:
    step = pm.NUTS()
    trace = pm.sample(4000, tune=2000, init=None, step=step, chains=2, cores=2)
    idata = az.from_pymc3(trace)
    az.plot_trace(idata, compact=True)
    output_bmtl = pd.DataFrame(az.summary(trace, round_to=2))#statistics of the estimated parameters
pm.model_to_graphviz(logreg_model)


#Predictions
pm.set_data({"X": x_test}, model=logreg_model)
pm.set_data({"Y": y_test}, model=logreg_model)
# Generate posterior samples.
ppc_test = pm.sample_posterior_predictive(trace, model=logreg_model, samples=1500)

y_mode = stats.mode(ppc_test['y_obs'])
y_pred = y_mode[0].reshape(900,3)

#Evaluation of the model:
y_test = y_test.to_numpy()
print(f"AUC score of the model: {roc_auc_score(y_test, y_pred)}")
print(f"AUC BMTL score FIBR_PRED: {roc_auc_score(y_test[:,0], y_pred[:,0])}")
print(f"AUC BMTL score ZSN: {roc_auc_score(y_test[:,1], y_pred[:,1])}")
print(f"AUC BMTL score LET_IS: {roc_auc_score(y_test[:,2], y_pred[:,2])}")
print(f'recall FIBR_PRED (TP/TP+FN): {recall_score(y_test[:,0], y_pred[:,0]): 0.3f}')
print(f'recall ZSN (TP/TP+FN): {recall_score(y_test[:,1], y_pred[:,1]): 0.3f}')
print(f'recall LET_IS (TP/TP+FN): {recall_score(y_test[:,2], y_pred[:,2]): 0.3f}')
print(f'precision FIBR_PRED (TP/TP+FP): {precision_score(y_test[:,0], y_pred[:,0]): 0.3f}')
print(f'precision ZSN (TP/TP+FP): {precision_score(y_test[:,1], y_pred[:,1]): 0.3f}/n')
print(f'precision LET_IS (TP/TP+FP): {precision_score(y_test[:,2], y_pred[:,2]): 0.3f}')