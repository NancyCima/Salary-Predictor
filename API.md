# Salary Prediction API

This API allows users to predict salaries based on demographic and professional data using a trained machine learning model and a FastAPI backend.

---

##  How to Use the API

### 1. Save Your Trained Model and Preprocessing Pipeline

```python
import joblib

joblib.dump(best_model, 'salary_model.pkl')
joblib.dump(full_pipeline, 'preprocessing_pipeline.pkl')
```

---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3. Run the API Locally

```bash
uvicorn app:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 4. Example Request Using `curl`
You can test it with:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 35,
    "gender": "Male",
    "education_level": "Master\'s",
    "job_title": "Data Scientist",
    "years_of_experience": 8,
    "description": "I am a data scientist with 8 years of experience in machine learning and statistical modeling."
}'
```

---

## API Documentation

Interactive API documentation is automatically available at:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Endpoints

| Method | Endpoint     | Description                          |
|--------|--------------|--------------------------------------|
| GET    | `/`          | Root information and available routes |
| GET    | `/health`    | Health check of the model             |
| POST   | `/predict`   | Predict salary with input JSON data  |

---

## Example Input JSON

```json
{
  "age": 35,
  "gender": "Male",
  "education_level": "Master's",
  "job_title": "Data Scientist",
  "years_of_experience": 8,
  "description": "I am a data scientist with 8 years of experience in machine learning and statistical modeling."
}
```

## Example Output

```json
{
  "predicted_salary": 90877.3,
  "confidence_interval": [
    81789.57,
    99965.03
  ]
}
```
```