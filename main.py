

from utils import load_boston_data, split_data, evaluate_model
from src.models import get_tuned_models  # or models.py if reused

df = load_boston_data()
X_train, X_test, y_train, y_test = split_data(df)

models = get_tuned_models(X_train, y_train)

for name, model in models.items():
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"{name}: MSE={mse:.2f}, R²={r2:.2f}")
