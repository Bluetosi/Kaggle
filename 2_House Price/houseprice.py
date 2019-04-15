import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# Origin
ori_train = pd.read_csv('./Input/train.csv')
ori_test = pd.read_csv('./Input/test.csv')

# Data Load
df_train = pd.read_csv('./Input/train.csv')
df_test = pd.read_csv('./Input/test.csv')

# 이상치 제거
df_train = df_train.loc[df_train['id']!=4123] # grade 3
df_train = df_train.loc[df_train['id']!=2775] # grade 11

# ID 제거
df_train.drop("id", axis=1, inplace=True)
df_test.drop("id", axis=1, inplace=True)

# 거래 년도 (라벨 인코딩)
df_train['year'] = df_train['date'].apply(lambda x: str(x[0:4])).astype(int)
df_test['year'] = df_test['date'].apply(lambda x: str(x[0:4])).astype(int)
le1 = LabelEncoder()
le1.fit(df_train['year'])
le1.fit(df_test['year'])
df_train['year'] = le1.transform(df_train['year'])
df_test['year'] = le1.transform(df_test['year'])

# 거래 년월 (라벨 인코딩)
df_train['yearmm'] = df_train['date'].apply(lambda x: str(x[0:6])).astype(int)
df_test['yearmm'] = df_test['date'].apply(lambda x: str(x[0:6])).astype(int)
le2 = LabelEncoder()
le2.fit(df_train['yearmm'])
le2.fit(df_test['yearmm'])
df_train['yearmm'] = le2.transform(df_train['yearmm'])
df_test['yearmm'] = le2.transform(df_test['yearmm'])

# 거래 날짜
df_train['date'] = df_train['date'].apply(lambda x: str(x[0:8])).astype(int)
df_test['date'] = df_test['date'].apply(lambda x: str(x[0:8])).astype(int)

# 우편번호 (라벨 인코딩)
le3 = LabelEncoder()
le3.fit(df_train['zipcode'])
le3.fit(df_test['zipcode'])
df_train['zipcode'] = le3.transform(df_train['zipcode'])
df_test['zipcode'] = le3.transform(df_test['zipcode'])

# 재건축 여부
df_train['is_renovated'] = df_train['yr_renovated'].map(lambda x: 1 if x > 0 else 0)
df_test['is_renovated'] = df_test['yr_renovated'].map(lambda x: 1 if x > 0 else 0)

# 최신 건축 년도
df_train['yr_renovated'] = np.maximum(df_train['yr_built'], df_train['yr_renovated'])
df_test['yr_renovated'] = np.maximum(df_test['yr_built'], df_test['yr_renovated'])

# 방의 총 갯수
df_train['totalrooms'] = df_train['bedrooms'] + df_train['bathrooms']
df_test['totalrooms'] = df_test['bedrooms'] + df_test['bathrooms']

# 층 별 주거공간
df_train['sqft_living_floor'] = df_train['sqft_above'] / df_train['floors']
df_test['sqft_living_floor'] = df_test['sqft_above'] / df_test['floors']

# 부지 대비 건물 면적 비
df_train['sqft_building_ratio'] = df_train['sqft_living_floor'] / df_train['sqft_lot']
df_test['sqft_building_ratio'] = df_test['sqft_living_floor'] / df_test['sqft_lot']

# 평균 대비 주거 공간 오차
df_train['living15_diff'] = df_train['sqft_living'] - df_train['sqft_living15']
df_test['living15_diff'] = df_test['sqft_living'] - df_test['sqft_living15']

# 평균 대비 부지 오차
df_train['lot15_diff'] = df_train['sqft_lot'] - df_train['sqft_lot15']
df_test['lot15_diff'] = df_test['sqft_lot'] - df_test['sqft_lot15']

# 위도 단순화
def category_lat(x):
    if x < 47.2:
        return 0    
    elif x < 47.3:
        return 1
    elif x < 47.4:
        return 2
    elif x < 47.5:
        return 3
    elif x < 47.6:
        return 4
    elif x < 47.7:
        return 5
    else:
        return 6
    
df_train['lat_cat'] = df_train['lat'].apply(category_lat)
df_test['lat_cat'] = df_test['lat'].apply(category_lat)

# 경도 단순화
def category_long(x):
    if x < -122.5:
        return 0    
    elif x < -122.4:
        return 1
    elif x < -122.3:
        return 2
    elif x < -122.2:
        return 3
    elif x < -122.1:
        return 4
    else:
        return 5
    
df_train['long_cat'] = df_train['long'].apply(category_long)
df_test['long_cat'] = df_test['long'].apply(category_long)

# 등급 단순화
def category_grade(x):
    if x < 4:
        return 1
    elif x < 7:
        return 2
    elif x < 9:
        return 3
    elif x < 11:
        return 4    
    else:
        return 5
    
df_train['grade_cat'] = df_train['grade'].apply(category_grade)
df_test['grade_cat'] = df_test['grade'].apply(category_grade)

# 외관 점수 (cat)
df_train['out_score_cat'] = (df_train['view']+1) * df_train['grade_cat']
df_test['out_score_cat'] = (df_test['view']+1) * df_test['grade_cat']

# 내관 점수 (cat)
df_train['in_score_cat'] = df_train['condition'] * df_train['grade_cat']
df_test['in_score_cat'] = df_test['condition'] * df_test['grade_cat']

# 총괄 점수 (cat)
df_train['total_score_cat'] = df_train['out_score_cat'] + df_train['in_score_cat']
df_test['total_score_cat'] = df_test['out_score_cat'] + df_test['in_score_cat']

# 방 점수 (cat)
df_train['totalrooms_score_cat'] = df_train['totalrooms'] * df_train['grade_cat']
df_test['totalrooms_score_cat'] = df_test['totalrooms'] * df_test['grade_cat']

# 정규화
skew_columns = ['sqft_living','sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
for col in skew_columns:
    df_train[col] = df_train[col].map(lambda x: np.log1p(x))
    df_test[col] = df_test[col].map(lambda x: np.log1p(x))
    
# 가격 정규화 (np.log1p)
df_train['price'] = df_train['price'].map(lambda x: np.log1p(x))
        
# Drop features
df_train.drop(['grade_cat'], axis=1, inplace=True)
df_test.drop(['grade_cat'], axis=1, inplace=True)    

# 값 나누기
Y_train = df_train['price']
Y_check = np.expm1(Y_train)
df_train.drop('price', axis=1, inplace=True)
X_train = df_train
X_test = df_test

# Cross validation strategy
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse
    
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    
# 모델 생성
gboost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, max_depth=4, max_features="sqrt", min_samples_leaf=15, min_samples_split=10, loss="huber", random_state=64)
model_xgb = xgb.XGBRegressor(n_estimators=2500, learning_rate=0.05, max_depth=4, objective="reg:linear", eval_metric='rmse', colsample_bytree=0.8, gamma=0.05, min_child_weight=1.8, reg_alpha=0.5, subsample=0.8, silent=1, random_state=64, nthread=-1)
model_lgb = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.015, max_depth=4, objective="regression", num_leaves=31, min_data_in_leaf=30, min_child_samples=20, boosting="gbdt", feature_fraction=0.9, bagging_freq=1, bagging_fraction=0.9, bagging_seed=11, metric='rmse', lambda_l1=0.1, nthread=4, random_state=64)

## 모델 학습
# GradientBoosting
gboost.fit(X_train, Y_train)
gboost_train_pred = np.expm1(gboost.predict(X_train))
gboost_pred = np.expm1(gboost.predict(X_test))
print("RMSE(GradientBoosting): {:.6f}".format(rmse(Y_check, gboost_train_pred)))

# xgboost
model_xgb.fit(X_train, Y_train)
xgb_train_pred = np.expm1(model_xgb.predict(X_train))
xgb_pred = np.expm1(model_xgb.predict(X_test))
print("RMSE(XGB):              {:.6f}".format(rmse(Y_check, xgb_train_pred)))

# lightGBM
model_lgb.fit(X_train, Y_train)
lgb_train_pred = np.expm1(model_lgb.predict(X_train))
lgb_pred = np.expm1(model_lgb.predict(X_test))
print("RMSE(LightGBM):         {:.6f}".format(rmse(Y_check, lgb_train_pred)))

# 데이터 조합 (앙상블)
ensemble_train_pred = gboost_train_pred * 0.7 + xgb_train_pred * 0.1 + lgb_train_pred * 0.2
ensemble_pred = gboost_pred * 0.7 + xgb_pred * 0.1 + lgb_pred * 0.2
print("RMSE(EnsembleModel):    {:.6f}".format(rmse(Y_check, ensemble_train_pred)))

submission = pd.read_csv('./Input/sample_submission.csv')
submission['price'] = ensemble_pred
submission.to_csv('./Output/submission_ensemble.csv', index=False)