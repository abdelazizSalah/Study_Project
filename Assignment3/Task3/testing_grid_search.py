from fine_tuning import * 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

if __name__ == "__main__":
    testing_svm()