from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def get_tuned_models(X_train, y_train):
    models = {}

    ridge_params = {'alpha': [0.01, 0.1, 1, 10], 'solver': ['auto', 'svd', 'cholesky']}
    ridge = GridSearchCV(Ridge(), ridge_params, cv=5)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge

    dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    dt = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'max_features': ['sqrt', 'log2']}
    rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    return models