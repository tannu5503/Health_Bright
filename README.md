# Health_Bright
# HealthBright: Personalized Health and Wellness Dashboard

**HealthBright** is an AI-powered web application designed to help users track and improve their health through personalized diet, exercise, disease prediction, mood-based recommendations, and more.

## 🚀 Features

- 🔐 User authentication (signup, login, password reset)
- 🧠 Disease prediction based on symptoms using machine learning
- 🍽️ Personalized diet recommendation based on BMI, allergy, preference, and chronic conditions
- 🏃 Exercise recommendation based on fitness goals
- 😌 Mood-based recommendations including music, tea, workouts, and affirmations
- 📊 Daily health tracker: calories, water, steps, sleep, heart rate, blood pressure, and mood
- 📅 Monthly health summaries and motivational feedback
- 📋 User profile management with contact info and chronic disease tracking
- 🧾 Meal builder using the Spoonacular API
- 🔔 Notification preferences and user settings

## 🛠 Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, Jinja2 templates
- **Database**: SQLite3
- **ML Models**: scikit-learn (pickle serialized)
- **Email**: Flask-Mail with Gmail SMTP
- **APIs**: HuggingFace (emotion detection), Spoonacular (recipe search)

## 🧠 Machine Learning Models Used

- **Disease Prediction**: Based on symptoms using decision tree classifier with filtering
- **Diet Recommendation**: Predicts optimal diet using BMI, allergies, preference, and chronic disease
- **Exercise Recommendation**: Suggests workouts based on exercise type and intensity
- **Recipe Info**: Predicts nutrition and recipe links using custom models

## 📂 Folder Structure

```
.
├── app.py
├── templates/
│   └── *.html (UI templates)
├── static/
│   └── *.css / *.js / images
├── dandss.csv (Disease & Symptom dataset with recommendations)
├── *.pkl (ML model files and encoders)
├── health_bright.db (SQLite database)
└── README.md
```

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/healthbright.git
cd healthbright
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

_Required packages include: Flask, Flask-Mail, scikit-learn, pandas, numpy, requests, werkzeug, etc._

### 3. Set Up Environment Variables

Replace `app.config` values in `app.py` with your actual Gmail and app password:

```python
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'
```

### 4. Run the Application

```bash
python app.py
```

Visit: `http://127.0.0.1:5000/`

## 📊 Notes

- Initial database and tables are automatically created on the first run.
- Passwords are securely hashed.
- Ensure API keys are valid and not rate-limited (Spoonacular & HuggingFace).

## 🤝 Contributing

Feel free to fork and submit pull requests. Suggestions, bug fixes, and feature requests are welcome.

## 📜 License

This project is open-source under the MIT License.
