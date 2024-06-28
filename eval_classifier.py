# %%
import mlflow
import pandas as pd
# %%
mlflow.set_experiment('clasificardor de pesos y alturas')
# %%
df_train = pd.read_csv('data/alturas-pesos-mils-train.csv')
df_test = pd.read_csv('data/alturas-pesos-mils-test.csv')
# %%
df_train
# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# %%
models_dict = [
    {'model': GaussianNB, 'model_name': 'GaussianNB'},
    {'model': LinearDiscriminantAnalysis, 'model_name': 'LinearDiscriminantAnalysis'},
    {'model': QuadraticDiscriminantAnalysis, 'model_name': 'QuadraticDiscriminantAnalysis'},
    {'model': QuadraticDiscriminantAnalysis, 'model_name': 'QuadraticDiscriminantAnalysis', 'params': {'reg_param': 0.5, 'store_covariance': True}},
    {'model': LogisticRegression, 'model_name': 'LogisticRegression'}
]
# %%
selected_model = models_dict[3]
selected_model
# %%
mlflow.start_run(run_name=selected_model['model_name'])
# %%
model = selected_model['model'](**selected_model['params'])
model
# %%
mlflow.log_params(selected_model['params'])
# %%
model.fit(
    df_train[['Peso', 'Altura']], 
    df_train[['Genero']]
)
# %%
print(model.classes_)

# %% Naive Bayes
model.theta_
model.var_
model.class_count_
# %% QDA
print('Medias')
print(model.means_)
print()
print('Cov Matrix')
print(model.covariance_)
# %%
# model.coef_, model.intercept_
# %%
train_acc = model.score(
    df_train[['Peso', 'Altura']], 
    df_train[['Genero']]
)
train_acc
# %%
test_acc = model.score(
    df_test[['Peso', 'Altura']], 
    df_test[['Genero']]
)
test_acc
# %%
mlflow.log_metrics(
    {'train_acc': train_acc, 'test_acc': test_acc}
)
# %%
mlflow.end_run()
# %%
