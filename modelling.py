import pandas as pd # For data reading and manipulation
import numpy as np 
import matplotlib.pyplot as plt # For visualizations
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, classification_report,accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import warnings
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
import joblib # saving intermediate results

warnings.filterwarnings('ignore')


## Feature selection and engineering

# Reading data
train=pd.read_csv('data/train_s3TEQDk.csv')
X_test=pd.read_csv('data/test_mSzZ8RL.csv').drop('ID',1)

X_train = train.drop(['Is_Lead','ID'],axis=1)
Y_train = train[['Is_Lead']]


# Creating binary features from feature with missing values indicating missing values
class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Credit_Product']):
        self.cols = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for col in self.cols:
            X[col+'_missing']=X[col].isna().astype(int)
        return X


# Converting categorical features to integers
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='label', handle_unknown='error'):
        self.encoding = encoding
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        """

        if self.encoding not in ['onehot', 'label']:
            template = ("encoding should be either 'onehot' or label, got %s")
            raise ValueError(template % self.handle_unknown)

        self.features=X.select_dtypes(object).columns.tolist()

        self._label_encoders_ = [LabelEncoder() for _ in range(len(self.features))]

        for i,f in enumerate(self.features):
            le = self._label_encoders_[i]
            Xi = X[f]
            le.fit(Xi)

        self.categories_ = [le.classes_ for le in self._label_encoders_]
        return self

    def transform(self, X):
        """Transform X using fit encoding."""
        for i,f in enumerate(self.features):
            X[f] = self._label_encoders_[i].transform(X[f])
        return X


# Capping outliers to Q3+1.5*IQR and Q1-1.5*IQR
class OutlierTreatment(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Avg_Account_Balance']):
        self.cols = columns

    def fit(self,X,y=None):
        self.bounds={}
        for col in self.cols:
            self.bounds[col+'_ul'] = X[col].quantile(0.75) + 1.5*(X[col].quantile(0.75)-X[col].quantile(0.25)) # Upper limit
            self.bounds[col+'_ll'] = X[col].quantile(0.25) - 1.5*(X[col].quantile(0.75)-X[col].quantile(0.25)) # Lower limit
        return self

    def transform(self,X):
        for col in self.cols:
            X[col]=np.where(X[col]>self.bounds[col+'_ul'],self.bounds[col+'_ul'],X[col])
            X[col]=np.where(X[col]<self.bounds[col+'_ll'],self.bounds[col+'_ll'],X[col])
        return X


# Centering numeric variables around their mean scaling by standard deviation
class StandardScaling(BaseEstimator, TransformerMixin):
    def __init__(self,columns=['Age','Vintage','Avg_Account_Balance']):
        self.cols=columns

    def fit(self,X,y=None):
        self.means=X[self.cols].mean()
        self.stds=X[self.cols].std()
        return self

    def transform(self,X):
        X[self.cols]=(X[self.cols]-self.means)/self.stds
        return X


pipeline = Pipeline([
    ('missing',MissingIndicator()),
    ('outliers',OutlierTreatment()),
    ('encoding',CategoricalEncoder()),
    ('standardizing',StandardScaling())
])
X_train=pipeline.fit_transform(X_train)
X_test=pipeline.transform(X_test)


# Function for computing evaluation metrics
def get_metrics(model,data,true,kind='_model'):
    metrics={}
    preds=model.predict(data)
    metrics['ROC_AUC']=roc_auc_score(true,model.predict_proba(data)[:,1])
    roc=plot_roc_curve(model,data,true)
    roc.ax_.set_title('ROC: AUC={0:0.2f}'.format(metrics['ROC_AUC']))
    plt.savefig(f'plots/ROC_curve{kind}.png')
    plt.show()
    print(classification_report(true,preds))
    metrics['F1_micro']=f1_score(true,preds,average='micro')
    metrics['F1_macro']=f1_score(true,preds,average='macro')
    metrics['F1']=f1_score(true,preds)
    metrics['accuracy']=accuracy_score(true,preds)
    precision, recall, thresholds=precision_recall_curve(true,model.predict_proba(data)[:,1])  
    metrics['PR_AUC'] = auc(recall,precision)
    pr=plot_precision_recall_curve(model,data,true)
    pr.ax_.set_title('Precision-Recall curve: AUC={0:0.2f}'.format(metrics['PR_AUC']))
    plt.savefig(f'plots/PR_curve{kind}.png')
    plt.show()
    return metrics


# Target encoding
def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + np.random.normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
            parts = []
            for tr_in, val_ind in skf.split(train_data,train_data[target_col].values):
                # divide data
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + 
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + np.random.normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + np.random.normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.iloc[:len(train_data),:], 
            all_encoded.iloc[len(train_data):,:])


# Out of fold mean encoding
X_train_encoded,X_test_encoded=mean_encode(train_data=pd.concat([X_train,Y_train.iloc[:,0]],axis=1), test_data=X_test, columns=['Region_Code','Occupation','Channel_Code','Credit_Product'],
                target_col='Is_Lead', reg_method='k_fold', alpha=5, add_random=False, rmean=0, rstd=0.1, folds=7)

X_train=pd.concat([X_train,X_train_encoded],axis=1).drop(['Region_Code','Occupation','Channel_Code','Credit_Product'],1)
X_test=pd.concat([X_test,X_test_encoded],axis=1).drop(['Region_Code','Occupation','Channel_Code','Credit_Product'],1)

# Stratified 5-fold CV
skf=StratifiedKFold(n_splits=5,shuffle=True, random_state=2021)

# Logistic regression
from sklearn.linear_model import LogisticRegression

params={'C':np.logspace(-1, 1, 4, endpoint = True, base = 10)}
logistic=GridSearchCV(LogisticRegression( class_weight='balanced', solver='liblinear', penalty='l2'),params, n_jobs=-1,
                cv=skf, scoring='roc_auc')
logistic.fit(X_train,Y_train.iloc[:,0].values)
print(logistic.best_score_,logistic.best_params_) # Best average score: 0.8541315155438103 {'C': 10.0}
metrics=get_metrics(logistic,X_train,Y_train.iloc[:,0].values,kind='_logistic')
print(metrics) # Results on whole train set
#{'ROC_AUC': 0.854185626143465, 'F1_micro': 0.8015749313256688, 'F1_macro': 0.7540834203020388, 'F1': 0.6460142297081457, 'accuracy': 0.8015749313256689, 'PR_AUC': 0.6940625268699401}

## SVM classification with 5-fold CV - taking too long
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
# Grid search on SVC
params={'C':np.logspace(-1, 1, 3, endpoint = True, base = 10),'kernel':['rbf']}
svm=GridSearchCV(SVC(C=0.5, class_weight='balanced', probability=True, kernel='linear'),params, n_jobs=-1,
                cv=skf, scoring='roc_auc')
svm.fit(X_train,Y_train.iloc[:,0].values)
print(svm.best_score_,svm.best_params_) # Best average score:
metrics=get_metrics(svm,X_train,Y_train.iloc[:,0].values,kind='_svm')
print(metrics)



## XGBoost with 5-fold CV
params = {
        'min_child_weight': [5], # min weight in a child: higher=>conservative
        'gamma': [5], # min loss reduction: higher=>conservative
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [4]
        }
xgclassifier=XGBClassifier(n_estimators=100,eval_metric='auc')
xgb = GridSearchCV(xgclassifier, params, n_jobs=1, cv=skf,
                    scoring='roc_auc')
xgb.fit(X_train,Y_train.iloc[:,0].values)
print(xgb.best_score_,xgb.best_params_) # Best average score:  0.8732991519144802\
#{'colsample_bytree': 0.8, 'gamma': 5, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8}

metrics=get_metrics(xgb,X_train,Y_train.iloc[:,0].values,kind='_xgb') # Metrics on whole train data
print(metrics) #{'ROC_AUC': 0.8787864749731314, 'F1_micro': 0.8626391291077424, 'F1_macro': 0.784462439175051,\
# 'F1': 0.6546548390067222, 'accuracy': 0.8626391291077424, 'PR_AUC': 0.7590310374769162}


## Multi-layer perceptron with 5-fold CV
params = {
    'hidden_layer_sizes': [(50,30,10)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.05], # Regularization
    'learning_rate': ['constant'],
}
mlpclf=MLPClassifier(max_iter=300,batch_size='auto',shuffle=True, random_state=2021, verbose=True,
                     early_stopping=False)
mlp = GridSearchCV(mlpclf, params, n_jobs=-1, cv=skf, scoring='roc_auc')
mlp.fit(X_train, Y_train.iloc[:,0].values) 
print(mlp.best_score_,mlp.best_params_) # Best average score: 0.8724520798070209
#{'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 30, 10), 'learning_rate': 'constant', 'solver': 'adam'}
metrics=get_metrics(mlp,X_train,Y_train.iloc[:,0].values,kind='_mlp') # Metrics on whole train data
print(metrics) # {'ROC_AUC': 0.8738893457789153, 'F1_micro': 0.8608810662325771, 'F1_macro': 0.7795893875660349,\
# 'F1': 0.6457329395305456, 'accuracy': 0.8608810662325771, 'PR_AUC': 0.750926704993776}



## Stacking all models and performing K-fold CV
skf=StratifiedKFold(n_splits=3,shuffle=True, random_state=2021)

estimators = [LogisticRegression(C=10,class_weight='balanced', solver='liblinear', penalty='l2'),
              XGBClassifier(n_estimators=100,eval_metric='auc', min_child_weight=5, gamma=5, subsample=0.8,
              colsample_bytree=0.8, max_depth=4),
              MLPClassifier(max_iter=300,batch_size='auto',shuffle=True, random_state=2021, verbose=False,
                     early_stopping=False, activation='relu', alpha=0.05, learning_rate='constant', solver='adam')]

sclf = StackingCVClassifier(classifiers= estimators , use_probas=True,
                          meta_classifier=XGBClassifier(), cv=skf)

scores=cross_val_score(sclf, X_train, Y_train.iloc[:,0], cv=skf, scoring='roc_auc')
print("Average ROC AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

stacked = sclf.fit(X_train, Y_train.iloc[:,0])
joblib.dump(stacked,'models/final_model.pkl') 
metrics=get_metrics(stacked,X_train,Y_train.iloc[:,0].values,kind='_stacked') # Metrics on whole train data
print(metrics)



## Making test predictions
submission=pd.read_csv('data/test_mSzZ8RL.csv')[['ID']]
submission['Is_Lead']=stacked.predict_proba(X_test)[:,1]

submission.to_csv('submission.csv',header=True,index=False) # Writing final submissions
