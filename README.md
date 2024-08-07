# Fastapi + MLflow + streamlit

Setup env. I hope I covered all.
```bash
pip install -r requirements.txt
```
# Start app
Go in the root dir and run these

Streamlit
```bash
streamlit run frontend/streamlit_main.py
```

FastAPI 
```
uvicorn backend.main:app
```

MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///db/bakckend.db
```

## Docker
- Mlflow: http://localhost:5000
- FastApi: http://localhost:8000/docs
- Streamlit: http://localhost:8501/

```bash
docker-compose build
docker-compose up
```
