# Placement & Salary Prediction System

A production-ready Flask web application that predicts student placement eligibility and expected salary packages (in LPA) based on academic metrics, projects, internships, hackathon participation, and technical/communication skills.

---

## Features

- **Multi-Step Form**: Clean, responsive 3-step evaluation workflow (Academic Record, Experience & Projects, Skills).
- **Dual ML Models**:
  - **Placement Classification**: Gradient Boosting model predicting placement likelihood.
  - **Salary Regression**: Ensemble Voting Regressor predicting annual compensation (LPA).
- **Production Architecture**:
  - Environment variable isolation (`.env` with secure `SECRET_KEY`).
  - Strict input validation and range bounds checking.
  - Robust error handling (400, 404, 500, 503).
  - Vercel serverless deployment configuration.

---

## Project Structure

```text
Placement_Prediction_Using_Machine-Learning-master/
├── static/
│   └── css/
│       └── style.css            # Unified production stylesheet
├── templates/
│   ├── home.html                # Landing page
│   ├── index.html               # Multi-step candidate evaluation form
│   ├── output.html              # Prediction results & recommendations
│   └── about.html               # Technical overview & feature details
├── .env.example                 # Template for environment configuration
├── .env                         # Local environment variables (git-ignored)
├── .gitignore                   # Version control ignore rules
├── app.py                       # Main Flask web application
├── model.pkl                    # Serialized placement prediction model
├── model1.pkl                   # Serialized salary prediction model
├── requirements.txt             # Pinned project dependencies
├── runtime.txt                  # Python runtime specification
└── vercel.json                  # Vercel deployment configuration
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11+ (recommended Python 3.11–3.13)
- pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template configuration and set your environment variables:
```bash
cp .env.example .env
```
Generate a secure random secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Paste this value into `.env`:
```ini
SECRET_KEY=your_generated_secret_key_here
FLASK_DEBUG=false
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Deployment (Vercel)

The repository includes `vercel.json` configured for `@vercel/python`. When deploying to Vercel:
1. Import the repository into your Vercel dashboard.
2. In Project Settings > Environment Variables, add:
   - `SECRET_KEY`: Your generated production secret key.
   - `FLASK_DEBUG`: `false`

---

## License

This project is licensed under the MIT License.
