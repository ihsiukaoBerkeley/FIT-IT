## Task 1: General Questions
1. What is your preferred language when building predictive models and why?

My preferred language for building predictive models is Python. The reason is that Python has several powerful packages for model building, such as scikit-learn, TensorFlow, and PyTorch. Moreover, packages like Numpy, Pandas, Matplotlib, Seaborn, and PySpark are useful for data manipulation and visualization.

2. Provide an example of when you used SQL to extract data.

During the shrimp farm automation project, I used SQL to extract data from AWS Timestream, which stored time series sensor readings from our POC site. I mainly performed SQL queries for two tasks: visualization and experiments. When constructing a dashboard using Grafana, I ran simple queries to extract historical data points for graphs and complex queries with aggregate functions to display averages, medians, and more. In addition, I used SQL to extract data to evaluate experiment results from tuning various equipment and environmental parameters.

3. Give an example of a situation where you disagreed upon an idea or solution design with a co-worker.  How did you handle the case?

My colleague and me disagreed about which backend platform to use for a project POC. I proposed building our platform using AWS services because they provided the highest customizability. On the other hand, my colleague believed we should use an existing SaaS solution because of its user-friendly UI and minimal maintenance requirements. We met to brainstorm the pros and cons of each option and aligned the decision with the client's requirements. Ultimately, we concluded that my colleague's solution was more suitable for that specific client. We also agreed to follow the same procedure for future projects to match different clients' requirements.

4. What are your greatest strengths and weaknesses and how will these affect your performance here?

My greatest strengths are my eagerness to acquire new knowledge and my willingness to listen to others. Staying updated with important papers, SOTA models, and industry news is crucial. My eagerness to learn ensures I keep up with these developments and enables me to bring innovative solutions to the team. My willingness to listen encourages me to collaborate with others and helps bring different perspectives to my work, improving the overall quality of the project.
One of my weaknesses is that I can get tunnel vision after working on a task for too long. I may forget to occasionally take my engineering glasses off and put on other perspectives. To mitigate this, I would take regular breaks and seek fresh perspectives from my colleagues to ensure my work aligns with the goal.

## Task 2: Python model development

### Intructions:
1. ensure the following files are in their respective file paths
- main.py
- data/val_set.csv
- data/test.csv
- models/scaler_balanced.joblib
- models/Gender_encoder_balanced.joblib
- models/Geography_encoder_balanced.joblib
- models/HasCrCard_encoder_balanced.joblib
- models/IsActiveMember_encoder_balanced.joblib
- models/MNN_best_balanced.keras
2. create a python 3.11 virtual environment
3. activate the virtual environment
4. install required packages with provided requirements.txt with pip3
5. run main.py with python3
6. check whether the following output files are created:
- output.csv
- ROC_Curve.png
- Confusion_Matrix.png
- Precision_Recall_Curve.png

### Explanations:
#### EDA:
train.csv contains 165034 samples and 14 columns. It does not contain null values. Three columns with string values (Surname, Geography, and Gender) contain spaces. The label "Exited" has an imbalance distribution with 130113 (79%) samples with 0 and 34921 (21%) with 1. Numerical features are also not normally distributed.
#### Preprocessing:
Columns with string values were lowercased and stripped of any white spaces. Since columns like id, CustomerId, and Surname had little use for training, they were dropped. train.csv was then shuffled and split into train and validation sets with a 7-3 ratio. Because of their non-normal distribution, numerical features were normalized instead of standardized. Categorical features were encoded based on their values. Normalization scaler and categorical feature encoders were first fitted on the train set and then used to transform both train and validation sets. To address the imbalance label issue, a new train set was created using an undersampler to balance the label.
#### Model Training:
Four types of models, logistic regression (baseline), random forest, XGBoost, and multi-layer neural network, were created. Each model was experimented on balanced and unbalanced train sets and evaluated on the validation set. The multi-layer neural network was also experimented on the un-normalized and un-encoded train set. It was done by discretizing numerical (continuous) features into bins and one-hot encoding them. Parameter tuning was performed on random forest, XGBoost, and multi-layer neural networks using grid search and cross-validation. The best-performing sets of parameters were used to create the "best" models for evaluations.
#### Model Evaluation:
Due to the imbalance issue, **F1 score** was selected for model evaluation and parameter tuning. The reason was that the accuracy of the majority label would dominate the metric. Using accuracy might also cause models to only capture the distribution of labels. F1 score provided a good balance between precision and recall. Since there was no clear business case for this assignment, a clear choice between precision or recall did not exist. Hence, F1 score was selected. Other metrics like AUC under ROC were also used for evaluation.
#### Evaluation Results & Discussion:
| Model  | Dataset | Precision | Recall | F1 | AUC |
| -------| ------- | --------- | ------ | ---| --- |
| Logistic Regression | Unbalanced&Normalized | 0.68 | 0.34 | 0.46 | 0.81 |
| Logistic Regression | Balanced&Normalized | 0.44 | 0.74 | 0.55 | 0.81 |
| Random Forest | Unbalanced&Normalized | 0.74 | 0.56 | 0.64 | 0.89 |
| Random Forest | Balanced&Normalized | 0.53 | 0.80 | 0.64 | 0.89 |
| XGBoost | Unbalanced&Normalized | 0.75 | 0.56 | 0.64 | 0.89 |
| XGBoost | Balanced&Normalized | 0.53 | 0.80 | 0.64 | 0.89 |
| Multi NN | Unbalanced&Normalized | 0.80 | 0.46 | 0.58 | 0.88 |
| **Multi NN** | Balanced&Normalized | 0.58 | 0.75 | 0.65 | 0.89 |
| Multi NN | Unbalanced&Un-normalized | 0.73 | 0.55 | 0.63 | 0.88 |
| Multi NN | Balanced&Un-normalized | 0.45 | 0.87 | 0.59 | 0.88 |

Based on the table above, random forest, XGBoost, and multi-layer neural networks performed the best with balanced and unbalanced train sets. However, an interesting phenomenon was models fitted on the balanced train set had lower precision but higher recall. Using a train set with balanced labels greatly improved the models' ability to identify all positive (Exited = 1) labels. This also showed that the imbalance inhibited the models' ability to identify positive labels as the majority label (Exited = 0) dominated the dataset. However, it should be noted that this improvement in the ability to identify positive labels did come at the cost of decreased accuracy in positive predictions (lower precision). In the real world, the choice might depend on which metrics (recall vs precision) the specific use case cares about. For this assignment, the choice was made to follow the industry standard practice and select the model fitted on the balanced train set. Hence, the multi-layer neural network fitted on the balanced and normalized train set was chosen since it had the highest F1 Score.

### Directory:

```bash
├── main.py		#main program to run
├── requirements.txt	#list of packages required to run main.py
├── sample_output.csv	#sample output csv
├── sample_Confusion_Matrix.png	#sample output image
├── sample_Precision_Recall_Curve.png	#sample output image
├── sample_ROC_Curve.png	#sample output image
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