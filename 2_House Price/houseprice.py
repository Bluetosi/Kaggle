import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb

# Origin
ori_train = pd.read_csv('./Input/train.csv')
ori_test = pd.read_csv('./Input/test.csv')

# Data Load
df_train = pd.read_csv('./Input/train.csv')
df_test = pd.read_csv('./Input/test.csv')

# 이상치 제거
df_train = df_train.loc[df_train['id']!=456] # grade
df_train = df_train.loc[df_train['id']!=2777] # grade
df_train = df_train.loc[df_train['id']!=7259] # grade
df_train = df_train.loc[df_train['id']!=8990] # sqft_living
df_train = df_train.loc[df_train['id']!=1231] # living_ratio
df_train = df_train.loc[df_train['id']!=12209] # living_ratio
df_train = df_train.loc[df_train['id']!=12781] # living_ratio

# ID 처리
df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

# 가격 정규화 (np.log1p)
df_train['price'] = df_train['price'].map(lambda x: np.log1p(x))

# 날짜 변경 (년.월.일)
df_train['date'] = df_train['date'].apply(lambda x: str(x[0:6])).astype(int)
df_test['date'] = df_test['date'].apply(lambda x: str(x[0:6])).astype(int)

# sqft_living 정규화 (np.log1p)
df_train['sqft_living'] = df_train['sqft_living'].map(lambda x: np.log1p(x))
df_test['sqft_living'] = df_test['sqft_living'].map(lambda x: np.log1p(x))

# sqft_lot 정규화 (np.log1p)
df_train['sqft_lot'] = df_train['sqft_lot'].map(lambda x: np.log1p(x))
df_test['sqft_lot'] = df_test['sqft_lot'].map(lambda x: np.log1p(x))

# sqft_above 정규화 (np.log1p)
df_train['sqft_above'] = df_train['sqft_above'].map(lambda x: np.log1p(x))
df_test['sqft_above'] = df_test['sqft_above'].map(lambda x: np.log1p(x))

# sqft_basement 정규화 (np.log1p)
df_train['sqft_basement'] = df_train['sqft_basement'].map(lambda x: np.log1p(x))
df_test['sqft_basement'] = df_test['sqft_basement'].map(lambda x: np.log1p(x))

# yr_renovated
df_train['is_renovated'] = df_train['yr_renovated'].map(lambda x: 1 if x > 0 else 0)
df_test['is_renovated'] = df_test['yr_renovated'].map(lambda x: 1 if x > 0 else 0)

## 데이터 생성

# 방의 총 갯수
df_train['totalrooms'] = df_train['bedrooms'] + df_train['bathrooms']
df_test['totalrooms'] = df_test['bedrooms'] + df_test['bathrooms']

# 방 면적 비율
df_train['sqft_room'] = np.round(df_train['sqft_living']/df_train['totalrooms'])
df_train['sqft_room'] = df_train['sqft_room'].map(lambda x: x if np.isfinite(x) else 0)
df_test['sqft_room'] = np.round(df_test['sqft_living']/df_test['totalrooms'])
df_test['sqft_room'] = df_test['sqft_room'].map(lambda x: x if np.isfinite(x) else 0)

# 주거 공간 활용 비율
df_train['living_ratio'] = df_train['sqft_living']/df_train['sqft_lot']
df_train['living_ratio'] = df_train['living_ratio'].map(lambda x: np.log1p(x))
df_test['living_ratio'] = df_test['sqft_living']/df_test['sqft_lot']
df_test['living_ratio'] = df_test['living_ratio'].map(lambda x: np.log1p(x))

# 집의 외형적 평가
df_train['grade_look'] = (df_train['view']+1)*df_train['condition']
df_test['grade_look'] = (df_test['view']+1)*df_test['condition']

# One-hot encoding
df_train = pd.get_dummies(df_train, columns=['waterfront'], prefix='waterfront')
#df_train = pd.get_dummies(df_train, columns=['view'], prefix='view')
df_test = pd.get_dummies(df_test, columns=['waterfront'], prefix='waterfront')
#df_test = pd.get_dummies(df_test, columns=['view'], prefix='view')

# Drop features
df_train.drop(['yr_renovated'], axis=1, inplace=True)
df_test.drop(['yr_renovated'], axis=1, inplace=True)

# 값 나누기
Y_train = df_train['price']
df_train.drop('price', axis=1, inplace=True)
X_train = df_train
X_test = df_test

# Cross validation strategy
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

def cv_score(model):
    print("Model CV score : {:.4f}".format(np.mean(cross_val_score(model, X_train, Y_train)), 
                                         kf=kfold))

def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse
    
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=[5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5], cv=kf))
ENet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=[0.0002, 0.0003, 0.0004, 0.0005], l1_ratio=[0.8, 0.85, 0.9, 0.95, 0.99, 1], cv=kf, random_state=42))
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features="sqrt", min_samples_leaf=15, min_samples_split=10, loss="huber", random_state=42)
model_xgb = xgb.XGBRegressor(random_state=42)
model_lgb = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.01, max_depth=4, objective='regression', num_leaves=31, min_data_in_leaf=30, min_child_samples=20, boosting="gbdt", feature_fraction=0.9, bagging_freq=1, bagging_fraction=0.9, bagging_seed=11, metric='rmse', lambda_l1=0.1, nthread=4, random_state=4950)
averaged_models = AveragingModels(models = (gboost, model_lgb))

score = rmsle_cv(model_xgb)
