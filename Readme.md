# Classification Task - JobAThon


This is the solution description for the Job-A-Thon held by AnalyticsVidya in May 2021. The description is divided into sections pertaining to *exploratory data analysis*, *feature engineering*, *modelling*, *further improvements*.<br/> 
Exploratory plots are provided in [**eda.ipynb**](eda.ipynb) notebook. Feature engineering and modelling processes are contained in [**modelling.py**](modelling.py) script.<br/>



## Exploratory Data Analysis

* Train data consists of 10 features and 2,45,725 entries 
    * 1 unique id column - one unique id for each entry<br/><br/>
    * Features 'Gender', 'Region_Code', 'Occupation', 'Channel_Code',
       'Credit_Product', 'Is_Active' are categorical
        * Region_Code is a high cardinality feature with 35 unique codes<br/><br/>
    * 3 features - 'Age', 'Vintage', 'Avg_Account_Balance' - are integer type
        * All are positively skewed<br/><br/>
    * 1 feature Credit_Product has missing values (29,325)
        * Values being missing is strongly associated with the dependent variable<br/><br/>
    * All 3 numeric features have outliers
        * Age has outliers in class 1
        * Vintage has outliers in class 0
        * Avg_Account_Balance has outliers in both categories<br/><br/>
    * Vintage and Age have a strong positive correlation (0.6-0.8), rest are very weakly correlated<br/><br/>
    * Ages are on average higher for class 1 than 0. Vintage shows different distributions for the two classes<br/><br/>
    * Occupation, Channel_Code, Credit_Product, Is_Active have apparent differences in their distributions with respect to the dependent variable (0/1)



* Dependent variable is imbalanced - 76% of the values are 0

## Feature Engineering


### Pre-processing and engineering
* 1 dummy feature is extracted from missing values of Credit_Product indicating whether the value is missing or not (0/1).
* 6 categorical variables are converted to integers using label encoding
* Outliers of Avg_Account_Balance are capped at upper and lower bounds of their respective features using the inter-quartile range (Q3+1.5*IQR and Q1-1.5*IQR).
* All non-categorical columns are centered around their mean and scaled by their standard deviation.
* Categorical features with cardinality > 5 are target mean-encoded using out of fold means with smoothing. Number of folds is 7, alpha is 5.

    

## Modelling

### Metrics for evaluation
* **ROC_AUC**: A balanced metric for evaluating performance on both positive and negative classes.
* **F1**: 2\**Precision*\**Recall*/(*Precission*+*Recall*). Harmonic mean of precision and recall of positive class.
* **F1-micro**: Overall F1 score  of all the classes put together. Reveals the aggregate performance of the model on all classes.
* **F1-macro**: Average of F1-scores all the classes. Can be low when one of the f1-scores is low.
* **Accuracy**: Proportion of samples correctly classified. Can be misleading in imbalanced dataset such as ours. 
* **Precision-Recall AUC**: Does not take True Negatives into account. It is more focused on the positive class and should be preferred
                            to ROC when positive class is of greater interest.<br/><br/>

### Machine learning algorithms used

An ensemble of linear, tree-based, and neural network models are employed to find relevant patterns in the data for 
effective supervised learning.
5-fold cross validation is performed for hyperparameter tuning in each model.
The ROC and PR curves are plotted for each model and saved in the **plots** directory.<br/>


* *Logistic regression*
    * Logistic regression with *l2* penalty and *balanced* class weights is performed on the whole train dataset, providing an average *ROC AUC* 0.8541315155438103<br/>
      Best hyperparater: *{'C': 10.0}*<br/>
      Results on the whole training set: *{'ROC_AUC': 0.854185626143465, 'F1_micro': 0.8015749313256688, 'F1_macro': 0.7540834203020388, 'F1': 0.6460142297081457, 'accuracy': 0.8015749313256689, 'PR_AUC': 0.6940625268699401}*<br/>  
      This gives a baseline indication of performance of linear models on the given data. <br/><br/>
* *XGBoost*
    * Extreme Gradient boosted trees using many weak learners to form a strong learner.
    * XGBoost uses first and second order gradients of the loss function to arrive at the minima quicker, and each node split is evaluated
      quickly using these gradients as they are constant for all possibles splits in a particular iteration. Regularization introduced to 
      the loss makes it less prone to overfitting.
    * 5-fold cross validation is performed on train for hyperparameters tuning and model evaluation. 
        * Best hyperparameters are: *{'colsample_bytree': 0.8, 'gamma': 5, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8}*
        * The best model yields an average ROC AUC of *0.873299*<br/><br/> 
* *Multi-layer Perceptron*
    * A neural network is a highly flexible model which can be fine tuned for a variety of tasks.
    * Hyperparameters are tuned on 5-fold cross validation 
        * Best hyperparameters are: *{'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 30, 10), 'learning_rate': 'constant', 'solver': 'adam'}*
        * Average ROC AUC of the best model is *0.8724521*.<br/><br/> 

* *Stacking*
    * A stacking classifier (XGBoost classifier) is trained on the outputs of these 3 models.
        * Evaluation is performed with 5-fold cross-validation.
        * 5-fold training and predictions taking too long.
        * Making predictions on 3-fold probabilty predictions.

    * Finally, predictions are made for the pre-processed test data and saved in the file [**submission.csv**](submission.csv).<br/><br/>


## Further possible improvements
* Grid search for hyperparameter tuning is time consuming and memory hogging. A better approach would be to use bayesian optimization to search in 
   areas that are likely to give better results. [*Hyperopt*](http://hyperopt.github.io/hyperopt/) is a good alternative. This can allow us to further
   tune our models.
* Hyperparameter optimization on the stacking classifier may yield better results. Muti-level stacking could also be beneficial.
* Introduction of interaction terms from existing variables may provide useful information to the model for learning deeper patterns.<br/><br/>


## Major libraries used
* *pandas*, *numpy* : Reading and processing data
* *sklearn*, *xgboost*: Data pre-processing and machine learning
* *matplotlib*, *seaborn*: Data visualization
* *statsmodels*: Statistical tests and analysis






