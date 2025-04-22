# Salary Prediction using Machine Learning

## Overview

This project presents a machine learning-based approach to predicting salaries based on a set of demographic and professional features. The dataset, sourced from a public repository, includes attributes such as education level, job title, experience, and geographical information. The primary objective is to construct a predictive model capable of estimating an individual’s salary using these input features.

## Problem Statement

The goal is to address the regression problem of salary prediction by leveraging supervised learning techniques. Accurate salary predictions can support HR analytics, job market analyses, and automated compensation recommendations.

## Project Structure

├── data/                    # Raw data files

├── src/

│   ├── preprocessing.py     # Data preprocessing pipeline

│   ├── model.py             # Model training logic

│   ├── evaluate.py          # Evaluation functions and metrics

│   ├── utils.py             # Helper functions

│   └── visualization.py     # Plotting utilities

├── Salary Prediction.ipynb  # Final Jupyter notebook

├── README.md

└── requirements.txt

Note that the code is modularized into scripts under src/ for clarity and reusability.

## Dataset

- **Source**: The dataset used for training the model includes anonymized records of individuals from multiple professional sectors. Data was loaded from three CSV files: people.csv, salary.csv, and descriptions.csv. These datasets were merged based on appropriate keys for a unified structure.

- **Features**: Includes both numerical and categorical variables, such as:

  - `Gender`

  - `Age`

  - `Education Level`

  - `Job Title`

  - `Years of Experience`

  - `Description`

- **Target Variable**: `Salary`

## Methodology

The following stages were implemented in the development process:

1. **Exploratory Data Analysis (EDA)**

 Initial data inspection was conducted to understand distributions, detect anomalies and outliers, and identify missing values. Visualizations and summary statistics were used to assess feature relevance and correlation with the target variable.

2. **Data Preprocessing**
- Missing values were removed.

- Categorical features were encoded using techniques such as One-Hot Encoding and OrdinalEncoder.
   
- Numerical features were standardized where appropriate.

- Feature engineering was used to:

        - Extract seniority level from job titles

        - Extract role family/category from job titles

        - Create an experience-education interaction feature

- The descriptions field was converted into semantic vectors for downstream ML with a pre-trained Sentence Transformer model.

3. **Model Training and Evaluation**

 The dataset was split into training and testing subsets. Model performance was assessed using metrics such as:

   - Mean Absolute Error (MAE)

   - Mean Squared Error (MSE)

   - R² Score

 The metrics were reported as intervals using a technique called bootstrapping. This works by doing resamples of the original prediction results (with replacement) many times (by default, 1000). Finally, it returns the 2.5th and 97.5th percentiles for each metric. This gives a 95% confidence interval, showing the expected variability in model performance.

 In addition, a Baseline vs. trained model performance comparison is done. A baseline is a simple model that sets the minimum performance bar. For regression, this is often a model that always predicts the mean or median of the target variable (like DummyRegressor in scikit-learn). Across all metrics, the trained model delivers a much stronger performance than the baseline. The improvement is both statistically significant and practically meaningful.

4. **Hyperparameter Tuning**

 To enhance model performance, an automated hyperparameter optimization was implemented using Optuna, a powerful framework for hyperparameter search. The objective was to fine-tune the `RandomForestRegressor` by exploring the optimal combination of:

    - `n_estimators`: Number of trees in the forest, searched in the range 100 to 1000.

    - `max_depth`: Maximum depth of each tree, searched in the range 3 to 15.

    - `min_samples_split`: Minimum number of samples required to split an internal node, searched in the range 2 to 20.

 The tuning process was guided by Optuna’s Tree-structured Parzen Estimator (TPE) algorithm, with performance evaluated using Negative Root Mean Squared Error (neg-RMSE) as the scoring metric. We applied 5-fold cross-validation over 25 optimization trials to ensure robust evaluation.

 Optuna was configured to maximize the mean neg-RMSE across the cross-validation folds—equivalent to minimizing the RMSE. Once the optimal hyperparameters were identified, the final model was retrained on the complete dataset using these best-found values.

5. **Model Interpretation**

 To gain insight into how the model makes predictions, two key interpretability techniques were employed:

- SHAP (SHapley Additive exPlanations) Values:

    SHAP was used to explain the impact of each feature on the model’s output. SHAP provides a unified measure of feature importance by quantifying how much each feature contributes—positively or negatively—to individual predictions. The resulting SHAP summary plot highlights the most influential features, helping to validate model behavior and uncover potential biases or unexpected patterns.

- Prediction vs Actual Plot:

    This scatter plot compares the model’s predicted salaries against the actual values from the dataset. It provides a visual assessment of model accuracy and generalization. A strong alignment along the diagonal line indicates that the model makes accurate predictions across the target range.

 Together, these tools enhance transparency and help validate that the model is accurate and interpretable.

## Results

The final model achieved satisfactory performance, demonstrating high predictive accuracy on the test set. The R² score and error metrics indicated a well-generalized model suitable for deployment in real-world scenarios.

## API

This project includes a fully functional REST API built with FastAPI for serving salary predictions. The API allows users to send demographic and professional data and receive a predicted salary along with a confidence interval.

### Endpoints

- `GET /`: Root endpoint providing API metadata and available routes.
- `GET /health`: Returns the health status of the model and confirms it is ready to serve predictions.
- `POST /predict`: Accepts user data and returns a salary prediction.
  
### Example Payload (POST /predict)
```json
{
  "age": 30,
  "gender": "Female",
  "education_level": "Master's",
  "job_title": "Data Scientist",
  "years_of_experience": 5,
  "description": "Experienced in machine learning and data analysis"
}
```

### Example Response
```json
{
  "predicted_salary": 88474.3,
  "confidence_interval": [
    79626.87,
    97321.73
  ]
}
```

The API also supports interactive documentation via Swagger UI at `/docs`.
More details are found in the 'API.md' file.

## Requirements

To run the notebook and reproduce the results, install dependencies via:

```bash
pip install -r requirements.txt
```

## Future Work

- Integration with a web interface for interactive predictions.
- Expansion of the dataset to enhance model generalization.

## License

This project is licensed under the MIT License.
