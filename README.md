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
├── README.md
├── assignment.docx
├── data
│   ├── test.csv
│   ├── train.csv
│   ├── train_norm_balanced.csv
│   ├── train_norm_set.csv
│   ├── train_set.csv
│   ├── train_set_balanced.csv
│   ├── val_norm_set.csv
│   └── val_set.csv
├── main.py
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
├── requirements.txt
└── src
    ├── EDA_preprocessing.ipynb
    ├── Logistic_Regression.ipynb
    ├── MNN_discret.ipynb
    ├── MNN_norm.ipynb
    ├── MNN_norm_param.ipynb
    ├── RandomForest.ipynb
    ├── SVM.ipynb
    └── XGBoost.ipynb
```