from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client
from dotenv import load_dotenv
import pickle, numpy as np, os
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "autoprice_secret_2024")

@app.after_request
def no_cache(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return r

# ── Supabase ──────────────────────────────────────────────────────────────────
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ── Auth ──────────────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id, name, email):
        self.id    = str(id)
        self.name  = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        r = sb.table("users").select("*").eq("id", user_id).single().execute()
        if r.data:
            return User(r.data["id"], r.data["name"], r.data["email"])
    except:
        pass
    return None

# ── Model Load ────────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)

model      = obj["model"]
model_name = obj["model_name"]
scaler     = obj["scaler"]
encoders   = obj["encoders"]
kmeans     = obj["kmeans"]
km_scaler  = obj["km_scaler"]
features   = obj["features"]
lwr_X      = obj["lwr_train_X"]
lwr_y      = obj["lwr_train_y"]

CAT_COLS = ["Make", "Model", "Transmission", "Fuel_Type", "Owner_Type"]

def lwr_predict_single(X_train, y_train, x, tau=1.5):
    diff    = X_train - x
    weights = np.exp(-np.sum(diff ** 2, axis=1) / (2 * tau ** 2))
    W       = np.diag(weights)
    X_b     = np.c_[np.ones(len(X_train)), X_train]
    x_b     = np.r_[1, x]
    theta   = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y_train.values)
    return float(x_b @ theta)

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]
        try:
            r = sb.table("users").select("*").eq("email", email).single().execute()
            if r.data and check_password_hash(r.data["password"], password):
                login_user(User(r.data["id"], r.data["name"], r.data["email"]))
                return redirect(url_for("dashboard"))
        except:
            pass
        flash("Invalid email or password.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name     = request.form["name"]
        email    = request.form["email"]
        password = request.form["password"]
        try:
            existing = sb.table("users").select("id").eq("email", email).execute()
            if existing.data:
                flash("Email already registered.")
                return render_template("signup.html")
            r = sb.table("users").insert({
                "name": name, "email": email,
                "password": generate_password_hash(password)
            }).execute()
            user = r.data[0]
            login_user(User(user["id"], user["name"], user["email"]))
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash("Signup failed. Try again.")
    return render_template("signup.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ── App Routes ────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    r       = sb.table("history").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
    history = r.data or []
    total   = len(history)
    avg     = round(sum(h["price"] for h in history) / total, 2) if total else 0
    highest = max((h["price"] for h in history), default=0)
    lowest  = min((h["price"] for h in history), default=0)
    recent  = history[:5]
    return render_template("dashboard.html", total=total, avg=avg,
                           highest=highest, lowest=lowest, recent=recent)

@app.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html")

@app.route("/history")
@login_required
def history_page():
    r       = sb.table("history").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
    history = r.data or []
    return render_template("history.html", history=history)

@app.route("/about")
def about():
    return render_template("about.html")

# ── Predict API ───────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.json
    try:
        row = {
            "Year":         int(data["year"]),
            "Make":         data["make"],
            "Model":        data["model"],
            "Engine_Size":  float(data["engine_size"]),
            "Mileage":      float(data["mileage"]),
            "Transmission": data["transmission"],
            "Fuel_Type":    data["fuel_type"],
            "Owner_Type":   data["owner_type"],
        }
        for col in CAT_COLS:
            enc = encoders[col]
            val = row[col]
            if val not in enc.classes_:
                return jsonify({"error": f"Unknown value '{val}' for {col}"}), 400
            row[col] = int(enc.transform([val])[0])

        km_input = km_scaler.transform([[row["Year"], row["Engine_Size"], row["Mileage"]]])
        row["Cluster"] = int(kmeans.predict(km_input)[0])

        x_arr    = np.array([[row[f] for f in features]])
        x_scaled = scaler.transform(x_arr)

        if model_name == "LWR":
            price_usd = lwr_predict_single(lwr_X, lwr_y, x_scaled[0])
        else:
            price_usd = float(model.predict(x_scaled)[0])

        price_inr = round(max(0, price_usd * 83.5), 2)

        sb.table("history").insert({
            "user_id":      current_user.id,
            "date":         datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "make":         data["make"],
            "model":        data["model"],
            "year":         data["year"],
            "mileage":      data["mileage"],
            "fuel_type":    data["fuel_type"],
            "transmission": data["transmission"],
            "price":        price_inr,
            "algorithm":    model_name,
        }).execute()

        return jsonify({"price": price_inr, "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
