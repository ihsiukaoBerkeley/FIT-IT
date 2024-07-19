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

|
├── test.py					# Main python program
├── requirements.txt		# Contains required packages name
├── src						# Contains working notebooks
│   ├── EDA_preprocessing	# Conducts EDA on train data, splits validation data, creates scaler │	│							encoder
│   ├── Logistic_Regression # Creates Logistic Reg
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...
