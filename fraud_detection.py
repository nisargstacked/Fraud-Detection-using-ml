"""Streamlit transaction fraud prediction demo.
Run from the project folder with: streamlit run fraud_detection.py
The app trains a starter model from AIML Dataset.csv on first launch if no saved
model exists. Training may take a few minutes; later launches reuse the saved model.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "fraud_detection_model.joblib"
DATA_PATH = HERE / "AIML Dataset.csv"

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("Fraud Detection Prediction App")
st.write("Enter the transaction details below, then select **Predict**.")
st.caption("Educational tutorial demo. Predictions are estimates from a starter model, not a production fraud decision.")

@st.cache_resource(show_spinner="Preparing the fraud detection model… this can take a few minutes on the first run.")
def get_model():
    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)
        return saved["pipeline"], saved["features"]
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(DATA_PATH)
    target = "isFraud"
    if target not in df.columns:
        raise ValueError(f"Expected a {target!r} column in {DATA_PATH.name}.")
    max_rows = 500_000
    if len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, random_state=42, stratify=df[target])

    excluded = {target, "isFlaggedFraud", "nameOrig", "nameDest"}
    features = [col for col in df.columns if col not in excluded]
    X, y = df[features], df[target]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=300,
                                           solver="saga", random_state=42, n_jobs=-1)),
    ])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    model.fit(X_train, y_train)
    joblib.dump({"pipeline": model, "target": target, "features": features}, MODEL_PATH)
    return model, features

with st.form("transaction_form"):
    st.subheader("Transaction details")
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
    step = st.number_input("Time step", min_value=1, value=1, step=1,
                           help="The dataset's step field records the transaction time period.")
    st.markdown("**Sender balances**")
    col_a, col_b = st.columns(2)
    old_org = col_a.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=100.0)
    new_org = col_b.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=100.0)
    st.markdown("**Receiver balances**")
    col_c, col_d = st.columns(2)
    old_dest = col_c.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
    new_dest = col_d.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        with st.spinner("Preparing prediction…"):
            model, feature_names = get_model()
            values = {"step": step, "type": transaction_type, "amount": amount,
                      "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
                      "oldbalanceDest": old_dest, "newbalanceDest": new_dest}
            transaction = pd.DataFrame([{name: values.get(name, None) for name in feature_names}])
            probability = float(model.predict_proba(transaction)[0, 1])
            prediction = int(model.predict(transaction)[0])
        if prediction == 1:
            st.error(f"Potential fraud detected — estimated fraud probability: {probability:.1%}")
        else:
            st.success(f"No fraud detected by this model — estimated fraud probability: {probability:.1%}")
        st.caption("A model prediction can be wrong. Review evaluation metrics before interpreting results.")
    except Exception as exc:
        st.error(f"Could not make a prediction: {exc}")
        st.info("Check that AIML Dataset.csv is beside fraud_detection.py and that required packages are installed.")
