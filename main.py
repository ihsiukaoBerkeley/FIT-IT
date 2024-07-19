import os
import logging

#Set absl logging level to suppress warnings
logging.getLogger('absl').setLevel(logging.ERROR)

#Standard Data Packages
import pandas as pd
import numpy as np

#Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

#tf and keras
import tensorflow as tf
import keras
from keras import models

#Scikit
from sklearn import metrics
from sklearn import preprocessing

#Other Packages
import joblib
import csv


def check_data(df):
    #check feature dimensions
    if df.shape[1] != 13:
        raise Exception("Incorrect Number of Features")

    #check for null values
    if test_df.isnull().values.any():
        print("Found Missing Values, dropping...")
        df = df.dropna()

    return df

def pre_process(df):
    #define features
    feature_names = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    quantitative_columns = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    categorical_columns = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]

    #making test set and selecting features
    df_features = df.copy(deep=True)
    df_features = df_features[feature_names]
    
    #clean and transform string features
    df_features["Geography"] = df_features["Geography"].str.strip()
    df_features["Geography"] = df_features["Geography"].apply(str.lower) 
    df_features["Gender"] = df_features["Gender"].str.strip()
    df_features["Gender"] = df_features["Gender"].apply(str.lower)
    
    #load scaler
    scaler = joblib.load("models/scaler_unbalanced.joblib")
        
    #transform validation data
    df_features[quantitative_columns] = scaler.transform(df_features[quantitative_columns])
    
    for i in categorical_columns:
        #load encoder
        encoder_name = f"models/{i}_encoder_unbalanced.joblib"
        label_encoder = joblib.load(encoder_name)
    
        #transform validation data
        df_features[i] = label_encoder.transform(df_features[i])

    df_features = df_features.astype("float64")

    return df_features

def make_predictions_RF(df, model_path):
    #load model
    model = joblib.load(model_path)
    
    #make predictions
    predictions = model.predict(df)
    
    #calculate label probability
    y_pred_proba = model.predict_proba(df)[:,1]

    return predictions, y_pred_proba
    
    

def make_predictions_MNN(df, model_path):
    #load model
    model = keras.saving.load_model(model_path)
    
    #make predictions
    predictions = model.predict({
        "CreditScore": df[["CreditScore"]],
        "Geography": df[["Geography"]],
        "Gender": df[["Gender"]],
        "Age": df[["Age"]],
        "Tenure": df[["Tenure"]],
        "Balance": df[["Balance"]],
        "NumOfProducts": df[["NumOfProducts"]],
        "HasCrCard": df[["HasCrCard"]],
        "IsActiveMember": df[["IsActiveMember"]],
        "EstimatedSalary": df[["EstimatedSalary"]],
    })
    
    return predictions

def plot_roc(actual_label, predictions):
    #calculate false positve and true positive rate
    fpr, tpr, _ = metrics.roc_curve(actual_label,  predictions)
    
    #calculate AUC
    auc = metrics.roc_auc_score(actual_label, predictions)

    #make plot
    plt.plot(fpr, tpr, label = "data 1, auc = " + str(auc))
    plt.legend(loc = 4)
    plt.title("ROC Curve of Validation Data")
    
    plt.savefig('ROC_Curve.png')

    return auc

def plot_confusion_matrix(cm):
    plt.figure(figsize=(9,9))

    #create heatmap using confusion matrix
    sns.heatmap(cm, annot = True, fmt = ".3f", linewidths = .5, square = True, cmap = "Blues_r")

    #make plot
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix", size = 15)
    
    plt.savefig('Confusion_Matrix.png')

def plot_pr_curve(actual_label, predictions):
    #calculate precision and recall
    precision, recall, thresholds = metrics.precision_recall_curve(actual_label, predictions)
    avg_precision = metrics.average_precision_score(actual_label, predictions)

    #make plot
    plt.figure(figsize=(9, 9))
    plt.plot(recall, precision, color='blue', label = f"Avgerage Precision = {avg_precision:.2f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    plt.savefig('Precision_Recall_Curve.png')

def evaluate_predictions_sklearn(actual_label, pred_label):
    #calculate evaluations
    report = metrics.classification_report(actual_label, pred_label, output_dict = True)

    return report

def evaluate_predictions_MNN(actual_label, predictions):
    #convert predictions to labels
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    #calculate evaluations
    report = metrics.classification_report(actual_label, predictions, output_dict = True)

    return report, predictions

if __name__ == "__main__":
    #load test data and preprocess it
    test_df = pd.read_csv("data/test.csv")
    test_df = check_data(test_df)
    test_features = pre_process(test_df)

    #load validation data and preprocess it
    val_df = pd.read_csv("data/val_set.csv")
    val_features = pre_process(val_df)

    #make predictions #rf_best_balanced.joblib
    #test_pred = make_predictions_MNN(test_features, "models/mnn_norm_unbalanced.keras")
    #val_pred = make_predictions_MNN(val_features, "models/mnn_norm_unbalanced.keras")
    test_pred_label, test_pred = make_predictions_RF(test_features, "models/rf_best_balanced.joblib")
    val_pred_label, val_pred = make_predictions_RF(val_features, "models/rf_best_balanced.joblib")

    
    #plot auc
    val_auc = plot_roc(val_df["Exited"],  val_pred)
    
    #evalute validation results
    #report, val_pred_label = evaluate_predictions_MNN(val_df["Exited"], val_pred)
    report = evaluate_predictions_sklearn(val_df["Exited"], val_pred_label)
    
    #plot pr curve
    plot_pr_curve(val_df["Exited"],  val_pred)
    
    #make confusion matrix
    confusion_matrix = metrics.confusion_matrix(val_df["Exited"], val_pred_label)
    plot_confusion_matrix(confusion_matrix)

    #output evaluation results to csv
    with open("output.csv", "w") as f:
        f.write("Model Evaluation\n")
        f.write("Exited,Precision,Recall,F-1_Score,Label_Count\n")
        f.write("0,%s,%s,%s,%s\n" % (report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"], report["0"]["support"]))
        f.write("1,%s,%s,%s,%s\n" % (report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], report["1"]["support"]))
        f.write("\n")
        f.write("AUC of Validation Data:,%s\n" % (val_auc))

    with open("output.csv", "a") as f:
        f.write("\n")
        f.write("Confusion Matrix\n")
        f.write("Actual Exited = 0,%s,%s\n"%(confusion_matrix[0][0], confusion_matrix[0][1]))
        f.write("Actual Exited = 1,%s,%s\n"%(confusion_matrix[1][0], confusion_matrix[1][1]))
        f.write(",Predicted Exited = 0,Predicted Exited = 1")

    #output test.csv predictions to csv
    test_pred[test_pred <= 0.5] = 0
    test_pred[test_pred > 0.5] = 1

    test_pred = test_pred.astype("int32")

    predict_test_df = pd.DataFrame(test_pred, columns = ["Exited"]) 
    
    output_df = pd.concat([test_df, predict_test_df], axis = 1)
    with open("output.csv", "a") as f:
        f.write("\n")
        f.write("test.csv Predictions\n")
        output_df.to_csv(f, index = False)

