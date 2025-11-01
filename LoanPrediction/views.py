from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import CSVUploadForm
import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define file paths
RESULTS_FILE = os.path.join(settings.MEDIA_ROOT, "last_model_results.pkl")
LAST_UPLOADED_FILE = os.path.join(settings.MEDIA_ROOT, "last_uploaded_file.pkl")
SCALER_PATH = os.path.join(settings.MEDIA_ROOT, "scaler.pkl")
MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "optimized_loan_model.pkl")

# Load previous model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def home(request):
    return render(request, 'website/index.html')

def aboutus(request):
    return render(request, 'website/aboutus.html')

def analysis(request):
    previous_results = load_previous_results() or {}

    accuracy = previous_results.get("accuracy")
    recall = previous_results.get("recall")
    f1_score_val = previous_results.get("f1_score")

    context = {
        'accuracy': accuracy * 100 if accuracy is not None else None,
        'recall': recall * 100 if recall is not None else None,
        'f1_score': f1_score_val * 100 if f1_score_val is not None else None,
        'last_accuracy': accuracy * 100 if accuracy is not None else None,
        'last_recall': recall * 100 if recall is not None else None,
        'last_f1_score': f1_score_val * 100 if f1_score_val is not None else None,
        'form': CSVUploadForm(),
    }

    return render(request, 'website/analysis.html', context)

def load_previous_results():
    """ Load previous model results if available """
    if os.path.exists(RESULTS_FILE):
        return joblib.load(RESULTS_FILE)
    return {"accuracy": None, "recall": None, "specificity": None}

def save_results(results):
    """ Save model results """
    joblib.dump(results, RESULTS_FILE)

def upload_csv(request):
    """ Handle file upload and model training """
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():
            csv_file = request.FILES.get("file")

            if csv_file:
                fs = FileSystemStorage(location=settings.MEDIA_ROOT)
                file_name = fs.save(csv_file.name, csv_file)
                file_path = fs.path(file_name)

                joblib.dump(file_path, LAST_UPLOADED_FILE)

                acc, precision, recall, f1, _ = train_model(file_path)

                save_results({
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

                messages.success(request, "✅ Model trained successfully!")

            else:
                if os.path.exists(LAST_UPLOADED_FILE):
                    file_path = joblib.load(LAST_UPLOADED_FILE)
                    acc, precision, recall, f1, _ = train_model(file_path)

                    save_results({
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    })

                    messages.info(request, "📌 Using previously uploaded file for analysis.")
                else:
                    messages.error(request, "❌ No previous file found. Please upload a dataset first.")
        else:
            messages.error(request, "❌ Invalid file format. Please upload a CSV file.")

    return redirect('analysis')

def preprocess_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.columns = df.columns.str.replace(' ', '')

    if 'loan_id' in df.columns:
        df = df.drop(['loan_id'], axis=1)

    df = pd.get_dummies(df)

    df.rename(columns={
        'education_ Graduate': 'education',
        'self_employed_ Yes': 'self_employed',
        'loan_status_ Approved': 'loan_status'
    }, inplace=True)

    df = df.drop(['education_ Not Graduate', 'self_employed_ No', 'loan_status_ Rejected'], axis=1)

    y = df['loan_status']
    X = df.drop(['loan_status'], axis=1)
    return X, y

def train_model(file_path):
    """ Train a Random Forest model and evaluate performance """
    X, y = preprocess_data(file_path)

    print("📊 Target Class Distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=0,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"🔍 Max predicted probability: {max(y_proba)*100:.2f}%")
    print(f"🔍 Min predicted probability: {min(y_proba)*100:.2f}%")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return acc, precision, recall, f1, None

def predict(request):
    """ Handle user input and predict loan eligibility """
    if request.method == "POST":
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return render(request, 'website/predict.html', {
                'error': "🚨 No trained model found! Please upload a dataset and analyze it first.",
                'hide_form': True
            })

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        try:
            user_data = pd.DataFrame([[ 
                float(request.POST['no_of_dependents']),
                float(request.POST['income_annum']),
                float(request.POST['loan_amount']),
                float(request.POST['loan_term']),
                float(request.POST['cibil_score']),
                float(request.POST['residential_assets_value']),
                float(request.POST['commercial_assets_value']),
                float(request.POST['luxury_assets_value']),
                float(request.POST['bank_asset_value']),
                1 if request.POST['education'] == 'Graduate' else 0,
                1 if request.POST['self_employed'] == 'Yes' else 0
            ]], columns=[
                'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                'luxury_assets_value', 'bank_asset_value', 'education', 'self_employed'
            ])

            user_data_scaled = scaler.transform(user_data)
            prediction_prob = model.predict_proba(user_data_scaled)[0][1] * 100

            approval_message = f"There is a {prediction_prob:.2f}% chance of getting approved."

            return render(request, 'website/predict.html', {
                'approval_message': approval_message,
                'prediction_prob': prediction_prob
            })

        except Exception as e:
            messages.error(request, f"❌ Error in prediction: {str(e)}")
            return render(request, 'website/predict.html', {'error': str(e)})

    return render(request, 'website/predict.html')
