from Assignment3.Task3.fine_tuning import *
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import numpy as np



def load_dataset_csv_rf(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)

    X = df.drop(columns=['label'])  # keep as DataFrame
    y = df['label'].values          # labels as np array

    # Encode categorical features
    X_encoded, _ = encode_categorical(X)

    # Encode labels
    y_encoded = LabelEncoder().fit_transform(y)

    return X_encoded, y_encoded


def load_dataset_knn(file_path):
    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    y = LabelEncoder().fit_transform(df['diagnosis'])
    X = df.drop(columns=['diagnosis'])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y


def load_dataset_svm(file_path):
    """
    Load and preprocess the SVM dataset.
    Drops ID and ZIP Code, extracts Personal Loan as label,
    and scales all numeric features.
    """
    df = pd.read_csv(file_path)

    # Drop useless columns
    df = df.drop(columns=["ID", "ZIP Code"])

    # Target variable
    y = df["Personal Loan"].values

    # Feature matrix
    X = df.drop(columns=["Personal Loan"])

    # SVM requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def load_elliptic_dataset(root_folder):
    healthy_folder = os.path.join(root_folder, "Healthy")
    broken_folder = os.path.join(root_folder, "BrokenTooth")

    X = []
    y = []

    # Healthy = 0
    for file in os.listdir(healthy_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(healthy_folder, file))
            X.append(df.values)     # append all rows
            y += [0] * len(df)      # one label per row

    # Broken = 1
    for file in os.listdir(broken_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(broken_folder, file))
            X.append(df.values)
            y += [1] * len(df)

    # concatenate all CSVs vertically
    X = np.vstack(X)
    y = np.array(y, dtype=int)

    return X, y



def create_dataset_one_class_svm():
    # Create an imbalanced dataset
    X, y = make_classification(n_samples=100000, n_features=2, n_informative=2,
                            n_redundant=0, n_repeated=0, n_classes=2,
                            n_clusters_per_class=1,
                            weights=[0.995, 0.005],
                            class_sep=0.5, random_state=0)

    # Convert the data from numpy array to a pandas dataframe
    df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

    # Check the target distribution
    df['target'].value_counts(normalize = True)
    return df[['feature1', 'feature2']], df['target'].values

def encode_categorical(X):
    X_encoded = X.copy()
    label_encoders = {}

    for col in X.columns:
        if X[col].dtype == 'object':  # column is categorical
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    return X_encoded, label_encoders







def testing_random_forest ():
    # load car_evaluation.csv
    print('Loading dataset...')
    X, y = load_dataset_csv_rf('random_forest_data.csv')

    # split into 60% train and 20% validation, and 20% test sets 
    print('Splitting dataset...')
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # Random Forest Grid Search
    
    rf = RandomForestClassifier(random_state=42)
    best_rf = grid_search_random_forest(rf, X_train, y_train, X_val, y_val, scoring_metric='f1_macro')
    print("Best Random Forest Model:", best_rf)

    # test set evaluation for Random Forest
    test_accuracy_rf = best_rf.score(X_test, y_test)
    print(f'Random Forest Test Accuracy: {test_accuracy_rf}')


def testing_knn():
    # load knn_data.csv
    print('Loading dataset...')
    X, y = load_dataset_knn('Knn_dataset.csv')

    # split into 60% train and 20% validation, and 20% test sets 
    print('Splitting dataset...')
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # KNN Grid Search
    knn = KNeighborsClassifier()
    best_knn = grid_search_knn(knn, X_train, y_train, X_val, y_val, scoring_metric='f1_macro')
    print("Best KNN Model:", best_knn)

    # test set evaluation for KNN
    test_accuracy_knn = best_knn.score(X_test, y_test)
    print(f'KNN Test Accuracy: {test_accuracy_knn}')


def testing_svm():
    # load svm_data.csv
    print('Loading dataset...')
    X, y = load_dataset_svm('SVM_dataset.csv')

    # split into 60% train and 20% validation, and 20% test sets 
    print('Splitting dataset...')
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # SVM Grid Search
    svm = SVC(random_state=42)
    best_svm = grid_search_svm(svm, X_train, y_train, X_val, y_val, scoring_metric='f1_macro')
    print("Best SVM Model:", best_svm)

    # test set evaluation for SVM
    test_accuracy_svm = best_svm.score(X_test, y_test)
    print(f'SVM Test Accuracy: {test_accuracy_svm}')


def ocsvm_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)



def testing_one_class_svm():
    print("Creating One-Class SVM dataset...")
    X, y = create_dataset_one_class_svm()  
    # IMPORTANT: y must be 0 (normal) or 1 (anomaly)

    print("Splitting dataset...")

    # TRAINING DATA = only normal samples
    X_train_normal = X[y == 0]
    y_train_normal = np.zeros(len(X_train_normal))   # OCSVM ignores labels but scorer needs it

    # FULL SPLIT FOR EVALUATION (normal + anomalies)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    # Grid Search for One-Class SVM
    ocsvm = OneClassSVM()
    best_ocsvm = grid_search_one_class_svm(
        ocsvm,
        X_train_normal,
        y_train_normal,
        X_val,
        y_val,
        scoring_metric='accuracy'  # using accuracy for simplicity
    )


    # -----------------------------
    # TEST SET EVALUATION
    # -----------------------------
    y_test_pred_raw = best_ocsvm.predict(X_test)
    y_test_pred = (y_test_pred_raw == -1).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Best One-Class SVM Model:", best_ocsvm)
    print(f"One-Class SVM Test Accuracy: {test_accuracy:.4f}")

    return best_ocsvm


def testing_elliptic_envelope():
    '''
    
        Best parameters: {'contamination': 0.05, 'support_fraction': None}
        Validation F1: 0.0429
        Validation Recall: 0.0230
        Validation Accuracy: 0.4890

        Elliptic Envelope Test Results:
        Recall:    0.0230
        F1 Score:  0.0429
        Accuracy:  0.4890
    
    '''
    print("Loading Elliptic Envelope dataset...")
    X, y = load_elliptic_dataset("Elliptic_data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train ONLY on healthy samples
    X_train_normal = X_train[y_train == 0]
    y_train_normal = y_train[y_train == 0]  # all zeros

    ee = EllipticEnvelope()

    best_ee = grid_search_elliptic_envelope(
        ee,
        X_train_normal,
        y_train_normal,  # dummy labels
        X_test,
        y_test,
        scoring_metric='accuracy'
    )

    # Test predictions
    y_pred_raw = best_ee.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)

    print("\nElliptic Envelope Test Results:")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")



if __name__ == "__main__":
    testing_elliptic_envelope()