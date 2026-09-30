import os
import logging
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, flash, redirect, url_for
import joblib

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# Security: Secret key for CSRF protection and sessions
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')
app.config['WTF_CSRF_ENABLED'] = True

# Column names must exactly match training order
PLACEMENT_COLS = [
    'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects',
    'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon',
    '12th Percentage', '10th Percentage', 'backlogs'
]
SALARY_COLS = PLACEMENT_COLS + ['PlacementStatus']

# Model file paths
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'model.pkl'))
SALARY_MODEL_PATH = os.environ.get('SALARY_MODEL_PATH', os.path.join(BASE_DIR, 'model1.pkl'))

# Lazy-load models to avoid memory duplication in multi-worker setups
_model = None
_model1 = None

def load_models():
    global _model, _model1
    if _model is None or _model1 is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _model1 = joblib.load(SALARY_MODEL_PATH)
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise RuntimeError(f"Model files missing. Ensure {MODEL_PATH} and {SALARY_MODEL_PATH} exist.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError("Model loading failed. Check file integrity.")
    return _model, _model1

def validate_input(value, min_val=None, max_val=None, field_name="field"):
    """Validate numeric input with bounds checking."""
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name}: invalid number")
    
    if min_val is not None and val < min_val:
        raise ValueError(f"{field_name}: must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"{field_name}: must be <= {max_val}")
    return val

def validate_required_radio(form, field_name, display_name):
    """Validate required radio button selection."""
    value = form.get(field_name)
    if value not in ('0', '1'):
        raise ValueError(f"{display_name}: selection required")
    return float(value)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    try:
        # Load models (lazy initialization)
        model, model1 = load_models()

        # --- Input validation ---
        name = request.form.get('name', 'Student').strip() or 'Student'
        
        # Academic fields with bounds
        cgpa = validate_input(request.form.get('cgpa'), 0, 10, "CGPA")
        twelfth_pct = validate_input(request.form.get('twelfth_pct'), 0, 100, "12th Percentage")
        tenth_pct = validate_input(request.form.get('tenth_pct'), 0, 100, "10th Percentage")
        backlogs = validate_input(request.form.get('backlogs'), 0, 50, "Backlogs")
        
        # Experience fields (radio buttons)
        internship = validate_required_radio(request.form, 'internship', "Internship")
        hackathon = validate_required_radio(request.form, 'hackathon', "Hackathon")
        
        # Project/certification counts
        major_projects = validate_input(request.form.get('major_projects'), 0, 20, "Major Projects")
        mini_projects = validate_input(request.form.get('mini_projects'), 0, 50, "Mini Projects")
        certifications = validate_input(request.form.get('certifications'), 0, 50, "Certifications")
        
        # Skills
        skills = validate_input(request.form.get('skills'), 0, 50, "Technical Skills")
        communication = validate_input(request.form.get('communication'), 0, 10, "Communication")

        # --- Build feature DataFrame ---
        feat_dict = {
            'CGPA': [cgpa],
            'Major Projects': [major_projects],
            'Workshops/Certificatios': [certifications],
            'Mini Projects': [mini_projects],
            'Skills': [skills],
            'Communication Skill Rating': [communication],
            'Internship': [internship],
            'Hackathon': [hackathon],
            '12th Percentage': [twelfth_pct],
            '10th Percentage': [tenth_pct],
            'backlogs': [backlogs],
        }

        df_feat = pd.DataFrame(feat_dict)[PLACEMENT_COLS]

        # --- Predictions ---
        placement_result = model.predict(df_feat)[0]
        is_placed = 1 if placement_result == 'Placed' else 0

        df_sal = df_feat.copy()
        df_sal['PlacementStatus'] = is_placed
        df_sal = df_sal[SALARY_COLS]

        salary_raw = model1.predict(df_sal)[0]
        salary_lpa = max(0.0, round(salary_raw / 100000, 2))
        salary_min = max(0.0, round(salary_lpa * 0.88, 2))
        salary_max = round(salary_lpa * 1.12, 2)

        logger.info(f"Prediction: name={name}, placed={is_placed}, salary_lpa={salary_lpa}, range=[{salary_min}, {salary_max}]")

        return render_template(
            'output.html',
            name=name,
            placed=(is_placed == 1),
            salary_lpa=salary_lpa,
            salary_min=salary_min,
            salary_max=salary_max,
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        flash(str(e), 'error')
        return render_template('index.html'), 400

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        flash("Service temporarily unavailable. Please try again later.", 'error')
        return render_template('index.html'), 503

    except Exception as e:
        logger.exception(f"Unexpected error during prediction: {e}")
        flash("An unexpected error occurred. Please try again.", 'error')
        return render_template('index.html'), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('home.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception(f"Server error: {e}")
    return render_template('home.html'), 500

if __name__ == '__main__':
    # Debug mode only for local development
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)