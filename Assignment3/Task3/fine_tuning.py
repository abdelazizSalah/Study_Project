from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# # Define the model
# rf = RandomForestClassifier(random_state=42)
def grid_search_random_forest(rf:RandomForestClassifier, X_train, y_train, X_val, y_val, scoring_metric = 'f1'):
    print("Starting Random Forest Grid Search...")
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Set up the grid search
    grid_search = GridSearchCV(
        estimator=rf,  # the machine learning model
        param_grid=param_grid, # the hyperparameters to tune
        scoring=scoring_metric, # evaluation metric -> give model that maximizes f1 score
        n_jobs=-1, # number of CPU cores to use, -1 means use all available cores
        
        )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Evaluate on validation set
    val_accuracy = best_rf.score(X_val, y_val)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Validation Accuracy: {val_accuracy}')

    return best_rf

def grid_search_svm(svm:SVC, X_train, y_train, X_val, y_val, scoring_metric = 'f1'):

    print("Starting SVM Grid Search...")

    # Define the model
    # svm = SVC(random_state=42)

    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # Set up the grid search
    grid_search = GridSearchCV(estimator=svm,
                               param_grid=param_grid,
                               scoring=scoring_metric,
                               
                               n_jobs=-1,
                               )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Evaluate on validation set
    val_accuracy = best_svm.score(X_val, y_val)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Validation Accuracy: {val_accuracy}')

    return best_svm


def grid_search_elliptic_envelope(X_train, y_train, X_val, y_val, scoring_metric = 'f1'):

    print("Starting Elliptic Envelope Grid Search...")

    # Define the model
    ee = EllipticEnvelope()

    # Define the parameter grid
    param_grid = {
        'support_fraction': [None, 0.5, 0.75],
        'contamination': [0.1, 0.2, 0.3]
    }

    # Set up the grid search
    grid_search = GridSearchCV(estimator=ee,
                               param_grid=param_grid,
                               scoring=scoring_metric,
                               
                               n_jobs=-1,
                               )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_ee = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_ee.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Validation Accuracy: {val_accuracy}')

    return best_ee


def grid_search_knn(knn:KNeighborsClassifier, X_train, y_train, X_val, y_val, scoring_metric = 'f1'):
    print("Starting KNN Grid Search...")
    # Define the model
    # knn = KNeighborsClassifier()

    # Define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Set up the grid search
    grid_search = GridSearchCV(estimator=knn,
                               param_grid=param_grid,
                               scoring=scoring_metric,
                               
                               n_jobs=-1,
                               )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_knn = grid_search.best_estimator_

    # Evaluate on validation set
    val_accuracy = best_knn.score(X_val, y_val)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Validation Accuracy: {val_accuracy}')

    return best_knn