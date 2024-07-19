### Exited Binary Classification

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
2. create a python 3.11 virtual environment: python3.11 -m venv env_name
3. activate the virtual environment: source env_name/bin/activate
4. install required packages with provided requirements.txt: pip3 install -r requirements.txt
5. run program: python3 test.py
6. check whether the following output files are created:
- output.csv
- ROC_Curve.png
- Confusion_Matrix.png
- Precision_Recall_Curve.png

### File Structures:

```bash
├── main.py		#main program to run
├── requirements.txt	#list of packages required to run main.py
├── src		#contains all working notebooks
│   ├── EDA_preprocessing	#conducts EDA on train, splits data, preprocesses data, balances dataset
│   ├── Logistic_Regression	#creates logistic regression models on unbalanced & balanced dataset
│   ├── MNN_discret	#creates multi-layers neural network by discretizing & encoding features into bins
│   ├── MNN_norm	#creates multi-layers neural network with normalized numeric features
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