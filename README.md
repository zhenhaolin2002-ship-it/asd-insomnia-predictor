# Insomnia Risk Predictor

Moderate-to-severe insomnia risk prediction tool for adolescents/young adults with ASD.  
Built with Logistic Regression + SHAP explanations, deployed via Streamlit.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app main code |
| `requirements.txt` | Python dependencies |
| `Best_Model_*.joblib` | Trained pipeline (upload manually, do NOT commit if data is sensitive) |

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click Deploy
