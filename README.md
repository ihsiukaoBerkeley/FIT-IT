## Exited Binary Classification (FIT IT)

### Intructions:
1. ensure the following files are in their respective file paths
- test.py
- data/val_set.csv
- data/test.csv
- models/scaler_unbalanced.joblib
- models/Gender_encoder_unbalanced.joblib
- models/Geography_encoder_unbalanced.joblib
- models/HasCrCard_encoder_unbalanced.joblib
- models/IsActiveMember_encoder_unbalanced.joblib
- models/mnn_norm_...tbd
2. create a python 3.11 virtual environment with command: python3.11 -m venv env_name
3. activate the virtual environment with command: source env_name/bin/activate
4. install required packages with provided requirements.txt with command: pip3 install -r requirements.txt
5. run the program with command: python3 test.py
6. check whether the following output files are created:
- output.csv
- ROC_Curve.png
- Confusion_Matrix.png
- Precision_Recall_Curve.png

### Explanations:
##### EDA:
train.csv contains 165034 samples and 14 columns. It does not contain null values. Three columns with string values (Surname, Geography, and Gender) contain spaces. The label "Exited" has an imbalance distribution with 130113 (79%) samples with 0 and 34921 (21%) with 1. Numerical features are also not normally distributed.
##### Preprocessing:
Columns with string values were lowercased and stripped of any white spaces. Since columns like id, CustomerId, and Surname had little use for training, they were dropped. train.csv was then shuffled and splited into train and validation sets with 7-3 ratio. Because of their non-normal distribution, numerical features were normalized instead of standarized. Categorical features were encoded based on their values. Normalization scaler and categorical feature encoders were first fitted on train set then used to transform both train and validation sets. To address the imbalance label issue, a new train set was created using undersampler to balance the label.
##### Model Training:
Four types of models, logistic regression, random forest, XGBoost, and multi-layer neural network were created. Each model was experimented on both balanced and unbalanced train set and evaluted on the validation set. Multi-layer neural network was also experimented on the un-normalzed and un-encoded train set. It was done by discretizing numerical (continuous) features into bins and one-hot encoding them. Parameter tuning was performed on random forest, XGBoost and multi-layer neural network using grid search and cross-validation. The best performing set of parameters were used to create the "best" models for evaluations
##### Model Evaluation:
Due to the imbalance issue, **F1 score** was selected for model evaluation and parameter tuning. The reason was that accuracy would be dominated by the accuracy of the majority label. Using accuracy might also cause models to only capture the distribution of labels. F1 score provided a good balance between precision and recall. Since there was no clear business case for this assignment, a clear choice between precision or recall did not exist. Hence, F1 score was selected. Other metrics like AUC under ROC were also used for evaluation.
##### Evaluation Results:


### File Structures:

```bash
├── main.py		#main program to run
├── requirements.txt	#list of packages required to run main.py
├── src		#contains all working notebooks
│   ├── EDA_preprocessing	#conducts EDA on train, splits data, preprocesses data, balances dataset
│   ├── Logistic_Regression	#creates logistic regression models on unbalanced & balanced dataset
│   ├── MNN_discret	#creates multi-layer neural network by discretizing & encoding features into bins
│   ├── MNN_norm	#creates multi-layer neural network with normalized numerical features
│   ├── MNN_norm_param		#param tuning for MNN using cross-validation
│   ├── RandomForest	#creates random forest models on unbalanced & balanced data, conducts param tuning
│   └── XGBoost	#creates XGBoost models on unbalanced & balanced data, conducts param tuning
├── data
│   ├── test.csv	#orginal data
│   ├── train.csv	#orginal data
│   ├── train_norm_balanced.csv	#normalized & balanced train data for training
│   ├── train_norm_set.csv #normalzed & unbalanced train data for training
│   ├── train_set.csv		#un-normalzied & unbalanced train data for training
│   ├── train_set_balanced.csv		#un-normalized & balanced train data for training
│   ├── val_norm_set.csv	#normalized validation data
│   └── val_set.csv	#un-normalized validation data
├── models
│   ├── Gender_encoder_balanced.joblib
│   ├── Gender_encoder_unbalanced.joblib
│   ├── Geography_encoder_balanced.joblib
│   ├── Geography_encoder_unbalanced.joblib
│   ├── HasCrCard_encoder_balanced.joblib
│   ├── HasCrCard_encoder_unbalanced.joblib
│   ├── IsActiveMember_encoder_balanced.joblib
│   ├── IsActiveMember_encoder_unbalanced.joblib
│   ├── MNN_best_unbalanced.keras
│   ├── logistic_balanced.joblib
│   ├── logistic_unbalanced.joblib
│   ├── mnn_discret_balanced.keras
│   ├── mnn_discret_unbalanced.keras
│   ├── mnn_norm_balanced.keras
│   ├── mnn_norm_unbalanced.keras
│   ├── rf_best_balanced.joblib
│   ├── rf_best_unbalanced.joblib
│   ├── scaler_balanced.joblib
│   ├── scaler_unbalanced.joblib
│   ├── xgb_best_balanced.joblib
│   └── xgb_best_unbalanced.joblib
├── README.md
└── assignment.docx
	
```