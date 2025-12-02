from pyexpat import features

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from Assignment3.Task2.labels_helper import decode_labels, encode_labels


def one_class_svm_train(X_train, X_test, y_train, y_test):
    ocsvm_clf = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05
        )
    )

    ocsvm_clf.fit(X_train)
    print("Model training done")
    joblib.dump(ocsvm_clf, "models/ocsvm.joblib")

    # binary labels: +1 normal, -1 anomal
    y_test_binary = np.where(y_test == 0, 1, -1)

    # --- 1) Scores für ROC-AUC ---
    scores = ocsvm_clf.decision_function(X_test)

    # ROC-AUC berechnen
    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # --- 2) Normale Klassifikation ---
    y_pred_numeric = ocsvm_clf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    # Accuracy lieber NICHT verwenden
    print("Accuracy (Don't use):", accuracy_score(y_test_binary, y_pred_numeric))

    return ocsvm_clf, roc_auc


#x - data, y - labels
def svm_train(X_train, y_train):


    # SVC-Klassifikator mit linearem Kernel erstellen
    # C ist der Regularisierungsparameter. Kleinere Werte bedeuten stärkere Regularisierung.
    svm_clf = SVC(kernel='linear', C=1.0, random_state=42)

    # Modell trainieren (fitten)
    svm_clf.fit(X_train, y_train)

    return svm_clf


def one_class_svm_predict():

    return


#indices not restored
#splits ds based on indices!
def split_training_and_test(ds, labels, train_indices_fold, test_indices_fold):

    # FEATURES AUFTEILEN
    X_train = ds[train_indices_fold]
    X_test = ds[test_indices_fold]

    # LABELS AUFTEILEN
    y_train = labels[train_indices_fold]
    y_test = labels[test_indices_fold]
    return X_train, X_test, y_train,y_test



#todo: debug
#train and test indices for each fold ([[][],...]
def test_svm(ds, labels, train_indices,test_indices, global_label_encoder):

    #first_fold_train_indices = train_indices[0]
    #first_fold_test_indices = test_indices[0]
    numeric_labels=encode_labels(global_label_encoder,labels)
    X_train, X_test, y_train, y_test=split_training_and_test(ds,numeric_labels, first_fold_train_indices, first_fold_test_indices)

    print(numeric_labels[len(numeric_labels)-10])
    X_test_last_500 = X_test[-500:]
    y_test_last_500 = y_test[-500:]
    print(X_test_last_500[1])
    print(y_test_last_500[1])
    one_class_svm_train(X_train[:1000], X_test, y_train[:1000], y_test)
    #for each fold:

    ocsvm_predict()
    return
