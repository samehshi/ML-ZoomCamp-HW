import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys

model_file = './uv_project/pipeline_v1.bin'
# ✅ LOAD YOUR MODEL AND DICTVECTORIZER
with open(model_file, 'rb') as f:
    dv,model = pickle.load(f)



class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/predict")
def predict(client: Lead):
    customer = client.dict()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    return {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        print("Running in Jupyter — start server with: !uvicorn app:app --host 0.0.0.0 --port 9696")
    else:
        uvicorn.run(app, host="0.0.0.0", port=9696)