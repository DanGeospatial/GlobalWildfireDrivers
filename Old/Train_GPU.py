"""
Set up for GPU testing
Calculate SHAP values on GPU for massive performance boost
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from math import sqrt
import xgboost as xgb
import pandas as pd
import shap
import numpy as np

v3_059 = ("Z:/Research/Wildfire Publications/SubarcticWildfireProbability/SubarcticWildfireProbabilityCode/Data"
          "/probability_wildfire_dataset.csv")

train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

print("Reading Dataset")
df = pd.read_csv(v3_059)

left_upper_rh = df['1_clm'].mul(17.269)
left_bottom_rh = df['1_clm'].add(273.3)
right_upper_rh = df['2_clm'].mul(17.269)
right_bottom_rh = df['2_clm'].add(237.3)

left_bracket_rh = left_upper_rh.div(left_bottom_rh)
right_bracket_rh = right_upper_rh.div(right_bottom_rh)

right_rh = left_bracket_rh.sub(right_bracket_rh)
rh = np.exp(right_rh)
rh_f = rh.to_frame('rh')

dfc = pd.concat([df, rh_f], axis=1)

y = dfc['prob']
x = dfc[["pre", "LC", "46_clm", "tpi", "43_clm", "34_clm", "45_clm", "VS", "WE", "24_clm", "37_clm", "49_clm",
         "Aspect", "33_clm", "39_clm", "38_clm", "40_clm", "50_clm", "7_clm", "32_clm", "44_clm", "36_clm", "rh",
         "DAH"]]

print("Splitting Data")
# train is now 60% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=32)

# test is now 20% of the initial data set
# validation is now 20% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=32)

print("Define XGBRegressor")

params = {
    'min_split_loss': 0,
    'learning_rate': 0.01,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.7059719473304258,
    'colsample_bytree': 0.8994514591758284,
    'colsample_bylevel': 0.8904199797234215,
    'colsample_bynode': 0.8754449733223327,
    'reg_lambda': 0.016916708991554074,
    'reg_alpha': 0.0011336609763898625,
    'random_state': 32,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',
    'sampling_method': 'gradient_based'
}

dtrain = xgb.DMatrix(x_train, y_train)
d_val = xgb.DMatrix(x_val, y_val)


# Input optimal parameters from search
# verbose_eval can crash JupyterLab - verbose_eval=False
regressor = xgb.train(params=params, dtrain=dtrain, num_boost_round=5000, early_stopping_rounds=10,
                      evals=[(dtrain, "train"), (d_val, "val")])

print("Save Model")
# regressor.save_model("D:/Wildfire_Results_v7/xgb_fine_tune.json")

predictions = regressor.predict(d_val)

# Calculate roc_auc_score
auc = roc_auc_score(y_val, predictions)
print('ROC AUC: ', auc)

"""
print("Run Explainer")
shap_values = regressor.predict(d_val, pred_contribs=True)

print("Format SHAP Values")
exp = shap.Explanation(shap_values[:, :-1], data=x_val, feature_names=x_val.columns)
exp_values = exp.values
print("Save SHAP Values")

# Export values to investigate
explain_export = pd.DataFrame({
    'row_id': x_val.index.values.repeat(x_val.shape[1]),
    'feature': x_val.columns.to_list() * x_val.shape[0],
    'feature_value': x_val.values.flatten(),
    'shap_values': exp_values.flatten()
})
explain_export.to_csv("gpu_shap_df_lc.csv")
"""
