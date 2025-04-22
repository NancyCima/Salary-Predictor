from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna

def train_model(X, y):
    """
        Train a RandomForestRegressor model using hyperparameter optimization with Optuna.

        This function performs hyperparameter tuning using Optuna to find the best combination 
        of parameters (`n_estimators`, `max_depth`, and `min_samples_split`) for a 
        RandomForestRegressor. It uses 5-fold cross-validation with negative root mean squared 
        error as the scoring metric. The model is trained on the dataset using the best 
        hyperparameters found during the optimization process.

        Parameters:
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix used for training.

        y : array-like or pandas.Series
            Target vector used for training.

        Returns:
        -------
        RandomForestRegressor
            A trained RandomForestRegressor model with optimized hyperparameters.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
        }

        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1) #Use n_jobs=-1 to take advantage of all cores of the CPU
        score = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_root_mean_squared_error'
        ).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    best_model = RandomForestRegressor(
        **study.best_params, random_state=42)
    return best_model.fit(X, y)