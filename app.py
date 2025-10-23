# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from models import db, User, HealthRecord, Appointment
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev_secret_change_me")

# Use an absolute path to the database inside the 'instance' folder
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'instance', 'healthtrack.sqlite')}"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

MODEL_PATH = "models_saved/model.pkl"
COLS_PATH = "models_saved/columns.pkl"
model = None
columns = None
if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLS_PATH)
    print("Model and columns loaded.")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_first_request
def create_tables():
    os.makedirs("instance", exist_ok=True)
    db.create_all()

@app.route("/")
def index():
    return render_template("index.html")

# Register
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"].lower().strip()
        password = request.form["password"]
        role = request.form.get("role", "patient")
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for("register"))
        user = User(name=name, email=email, password=generate_password_hash(password), role=role)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower().strip()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            session['user_id'] = user.id   # ✅ add this line
            return redirect(url_for("dashboard"))

        flash("Invalid credentials.", "danger")

    return render_template("login.html")

# Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

# Dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id

    if current_user.role == "doctor":
        # Doctor dashboard
        
        # 1. Appointments are fetched correctly here:
        appointments = Appointment.query.filter_by(doctor_id=user_id).all()
        
        # 2. You can keep this loop if you want to use the patient_records 
        #    list later for something, but it's not needed for the core fix.
        #    However, to make the template work, we only need 'appointments'.
        
        # 3. CRUCIAL FIX: Pass the 'appointments' list to the template!
        return render_template("dashboard_doctor.html", appointments=appointments) 

    else:
        # Patient dashboard (This section is already correct)
        health_records = HealthRecord.query.filter_by(user_id=user_id).all()
        low_risk = sum(1 for h in health_records if h.prediction < 0.5)
        high_risk = sum(1 for h in health_records if h.prediction >= 0.5)
        appointments = Appointment.query.filter_by(patient_id=user_id).all()
        doctors = User.query.filter_by(role='doctor').all()

        return render_template(
            'dashboard.html',
            health_records=health_records,
            appointments=appointments,
            doctors=doctors,
            low_risk_count=low_risk,
            high_risk_count=high_risk
        )

# Prediction form
@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if request.method == "POST":
        form = request.form.to_dict()
        # Convert numbers
        for k,v in form.items():
            try:
                form[k] = float(v)
            except:
                pass
        # Prepare DataFrame consistent with training columns
        if columns is None or model is None:
            flash("Prediction model not ready. Train model first.", "danger")
            return redirect(url_for("dashboard"))
        df = pd.DataFrame([form])
        df = pd.get_dummies(df)
        # ensure all columns exist
        for c in columns:
            if c not in df.columns:
                df[c] = 0
        df = df[columns]
        prob = float(model.predict_proba(df)[:,1][0])
        # save health data
        hd = HealthRecord(user_id=current_user.id, data_json=json.dumps(form), prediction=prob)
        db.session.add(hd)
        db.session.commit()
        threshold = 0.5
        return render_template("predict.html", prob=prob, threshold=threshold)
    # GET => show form
    # NOTE: The form fields must match your dataset features used in training
    return render_template("predict_form.html")

# Book appointment
@app.route("/book", methods=["POST"])
@login_required
def book():
    # Ensure only patients can book
    if current_user.role != "patient":
        flash("Only patients can book appointments.", "warning")
        return redirect(url_for("dashboard"))

    # Extract form data
    doctor_id = request.form.get("doctor_id")
    date_str = request.form.get("date")
    time_str = request.form.get("time")

    # Validate input
    if not (doctor_id and date_str and time_str):
        flash("Please select doctor, date, and time.", "warning")
        return redirect(url_for("dashboard"))

    try:
        doctor_id = int(doctor_id)
        date_obj = datetime.fromisoformat(date_str).date()
    except ValueError:
        flash("Invalid date or doctor selection.", "danger")
        return redirect(url_for("dashboard"))

    # Create and save appointment
    appointment = Appointment(
        patient_id=current_user.id,
        doctor_id=doctor_id,
        date=date_obj,
        time=time_str,
        status="pending"
    )
    db.session.add(appointment)
    db.session.commit()

    flash("Appointment requested successfully. Awaiting confirmation from doctor.", "success")
    return redirect(url_for("dashboard"))


# Doctor confirms
@app.route("/appointments/<int:appt_id>/confirm")
@login_required
def confirm(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if current_user.role != "doctor" or appt.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for("dashboard"))
    appt.status = "confirmed"
    db.session.commit()
    flash("Appointment confirmed.", "success")
    return redirect(url_for("dashboard"))

# Doctor can decline (optional)
@app.route("/appointments/<int:appt_id>/decline")
@login_required
def decline(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    if current_user.role != "doctor" or appt.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for("dashboard"))
    appt.status = "declined"
    db.session.commit()
    flash("Appointment declined.", "info")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
