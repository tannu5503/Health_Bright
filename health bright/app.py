from flask import Flask, render_template, request, redirect,  jsonify, url_for, flash, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from flask import session
from flask_mail import Mail, Message
import secrets
import sqlite3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests 
import random

from datetime import datetime

# Initialize app and configure extensions (single instance)
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'

mail = Mail(app)

API_KEY = "YOUR_API_KEY"
SPOONACULAR_ENDPOINT = "YOUR_API_KEY"

# Global dictionary for reset tokens
reset_tokens = {}

# Load models and encoders
with open("predict_diet.pkl", "rb") as diet_model_file:
    diet_model = pickle.load(diet_model_file)
with open("encoder_bmi.pkl", "rb") as bmi_encoder_file:
    encoder_bmi = pickle.load(bmi_encoder_file)
with open("encoder_allergy.pkl", "rb") as allergy_encoder_file:
    encoder_allergy = pickle.load(allergy_encoder_file)
with open("encoder_diet.pkl", "rb") as diet_encoder_file:
    encoder_diet = pickle.load(diet_encoder_file)
with open("encoder_preference.pkl", "rb") as preference_encoder_file:
    encoder_preference = pickle.load(preference_encoder_file)
with open("encoder_chronic.pkl", "rb") as chronic_encoder_file:
    encoder_chronic = pickle.load(chronic_encoder_file)



# Load exercise recommendation model
with open("predict_exercise.pkl", "rb") as exercise_model_file:
    exercise_model = pickle.load(exercise_model_file)
with open("encoder_exercise_type.pkl", "rb") as exercise_type_encoder_file:
    encoder_exercise_type = pickle.load(exercise_type_encoder_file)
with open("encoder_exercise_name.pkl", "rb") as exercise_name_encoder_file:
    encoder_exercise_name = pickle.load(exercise_name_encoder_file)
with open("encoder_exercise_intensity.pkl", "rb") as exercise_intensity_encoder_file:
    encoder_exercise_intensity = pickle.load(exercise_intensity_encoder_file)


# Load recipe models & encoder
with open('recipe_models.pkl', 'rb') as f:
    models = pickle.load(f)
with open('food_encoder.pkl', 'rb') as f:
    food_encoder = pickle.load(f)

# Load disease prediction model and encoders
with open('disease_model.pkl', 'rb') as f:
    disease_model = pickle.load(f)
with open('disease_encoder.pkl', 'rb') as f:
    disease_encoder = pickle.load(f)
with open('symptom_columns.pkl', 'rb') as f:
    symptom_columns = pickle.load(f)
# Load dandss.csv for diet, precaution, exercise, etc.
dandss = pd.read_csv('dandss.csv')



#PREDICTION FUNCTIONS
def predict_with_threshold(symptoms):
    symptoms = [s.lower().strip().replace(' ', '').replace('_', '') for s in symptoms if s]

    if len(symptoms) < 3:
        raise ValueError("Please provide at least 3 symptoms.")

    # Load necessary files
    with open('disease_model_threshold.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('disease_encoder_threshold.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('symptom_columns_threshold.pkl', 'rb') as f:
        all_symptoms = pickle.load(f)

    dandss = pd.read_csv('dandss.csv')
    dandss['Symptoms'] = dandss['Symptoms'].str.lower().str.replace(' ', '').str.replace('_', '')

    # Filter diseases — only keep diseases with at least 3 symptoms matching user symptoms
    eligible_diseases = []
    for _, row in dandss.iterrows():
        disease_symptoms = set(row['Symptoms'].split(','))
        matching_symptoms = disease_symptoms.intersection(symptoms)

        if len(matching_symptoms) >= 3:
            eligible_diseases.append(row['Disease'])

    if not eligible_diseases:
        raise ValueError("No disease found with at least 3 matching symptoms.")

    # Re-train model only on these diseases (on-the-fly filtering)
    filtered_df = dandss[dandss['Disease'].isin(eligible_diseases)]

    # Rebuild multi-hot for filtered data
    for symptom in all_symptoms:
        filtered_df[symptom] = filtered_df['Symptoms'].apply(lambda x: 1 if symptom in x else 0)

    # Encode diseases (in filtered scope)
    le = LabelEncoder()
    filtered_df['Disease'] = le.fit_transform(filtered_df['Disease'])

    X = filtered_df[all_symptoms]
    y = filtered_df['Disease']

    # Train temp model on this filtered data
    temp_model = DecisionTreeClassifier()
    temp_model.fit(X, y)

    # Build input vector from user symptoms
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    # Predict disease
    disease_encoded = temp_model.predict(input_df)[0]
    predicted_disease = le.inverse_transform([disease_encoded])[0]

    # Fetch diet, precautions, exercise from `dandss.csv`
    disease_info = dandss[dandss['Disease'].str.lower() == predicted_disease.lower()].iloc[0]
    return {
        "disease": predicted_disease,
        "diet": disease_info['Dietary Recommendations'],
        "precaution": disease_info['Precautions'],
        "exercise": disease_info['Yoga & Exercises'],
        "exercise_precaution": disease_info['Exercise Precautions']
    }


#PREDICTION RECIPES FUNCTION

def get_recipe_details(food_item):
    encoded_item = food_encoder.transform([food_item])[0]
    encoded_df = pd.DataFrame([[encoded_item]], columns=['Food_Item_Encoded'])

    details = {}
    for key, model in models.items():
        details[key] = model.predict(encoded_df)[0]

    return details

#PREDCICTION EXERCISE FUNCTION

def predict_exercise(exercise_type):
    df = pd.read_csv("full_exercise_recommendation.csv")
    
    # Filter dataset based on selected exercise type
    filtered_exercises = df[df["Exercise_Type"] == exercise_type]
    
    if filtered_exercises.empty:
        return []
    
    # Select up to 12 random exercises
    selected_exercises = filtered_exercises.sample(n=min(12, len(filtered_exercises)))
    
    # Format exercises for display
    exercise_list = []
    for _, row in selected_exercises.iterrows():
        exercise_list.append({
            "name": row["Exercise_Name"],
            "duration": row["Duration"],  # Fetching duration from dataset
            "intensity": row["Intensity"],
            "steps": row["Steps"],  # Fetching steps from dataset
            "link": row.get("Video_Link", "#") if pd.notna(row.get("Video_Link", "#")) else "#"

        })
    
    return exercise_list



#PREDICTION OF DIET

def predict_diet(height, weight, allergy, preference):
    bmi = weight / ((height / 100) ** 2)

    # Encode BMI category
    bmi_category = encoder_bmi.transform([
        "Underweight" if bmi < 18.5 else "Normal weight" if bmi <= 24.9 else "Overweight" if bmi <= 29.9 else "Obese"
    ])[0]

    # Encode allergy
    allergy_val = allergy.capitalize()
    if allergy_val not in encoder_allergy.classes_:
        allergy_val = "None" if "None" in encoder_allergy.classes_ else encoder_allergy.classes_[0]
    allergy_encoded = encoder_allergy.transform([allergy_val])[0]

    # Encode dietary preference
    preference_val = preference.capitalize()
    if preference_val not in encoder_preference.classes_:
        preference_val = "Vegetarian" if "Vegetarian" in encoder_preference.classes_ else encoder_preference.classes_[0]
    preference_encoded = encoder_preference.transform([preference_val])[0]

    # Fetch chronic disease from the database
    conn = sqlite3.connect("health_bright.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chronic_diseases FROM personal_info WHERE user_id = ?", (session["user_id"],))
    chronic_disease = cursor.fetchone()
    conn.close()

    # Process chronic disease
    chronic_disease = chronic_disease[0] if chronic_disease else "No Chronic Disease"

    # Encode Chronic Disease
    if chronic_disease not in encoder_chronic.classes_:
        chronic_disease = "No Chronic Disease"
    chronic_disease_encoded = encoder_chronic.transform([chronic_disease])[0]

    # Prepare input for the model
    input_data = pd.DataFrame([[height, weight, bmi_category, allergy_encoded, preference_encoded, chronic_disease_encoded]],
                              columns=["Height", "Weight", "BMI_Category", "Allergy", "Dietary_Preference", "Chronic_Disease"])

    # Predict diet
    diet_prediction = diet_model.predict(input_data)
    predicted_diet = encoder_diet.inverse_transform(diet_prediction)[0]

    # Debugging print statements
    print(f"Input Data: \n{input_data}")
    print(f"Predicted Diet: {predicted_diet}")

    return predicted_diet

#MOOD RECOMMENDATIONS
emotion_recommendations = {
    'joy': {
        'music': ["'Happy Vibes' playlist", "'Feel Good Classics'", "'Upbeat Indie Pop'"],
        'breathing': ["Box Breathing - 4-4-4-4 pattern"],
        'tea': ["Chamomile tea with honey 🍯"],
        'workout': ["10-minute Yoga for Stress Relief"],
        'podcast': ["The Mindful Minute"],
        'affirmation': ["Let your light shine. Today is yours to enjoy."]
    },
    'anger': {
        'music': ["'Calm Piano' playlist", "'Soothing Instrumentals'"],
        'breathing': ["4-7-8 Relaxing Breath"],
        'tea': ["Peppermint tea & a banana 🍌"],
        'workout': ["Light Walk & Stretch combo"],
        'podcast': ["Meditative Story"],
        'affirmation':  ["Breathe. You are in control of your response."]
    },
    'sadness': {
        'music': ["'Cheer Up' playlist", "'Soft Lofi Chill'"],
        'breathing': ["Deep Rhythmic Breathing"],
        'tea': ["Lavender & Lemon Balm tea 🍋"],
        'workout': ["5-minute Stretch Routine"],
        'podcast': ["Kind World"],
        'affirmation':  ["You are allowed to feel, and you are not alone."]
    },
    'fear': {
        'music': ["'Safe and Sound' playlist"],
        'breathing': ["Grounding Technique - 5 senses"],
        'tea': ["Holy Basil Tea 🍃"],
        'workout': ["Guided Meditation"],
        'podcast': ["The Daily Calm"],
        'affirmation': ["You are safe. This feeling will pass."]
    },
    'surprise': {
        'music': ["'Epic Chillwave Mix'", "'Unexpected Anthems'"],
        'breathing': ["Centering Breathwork"],
        'tea': ["Ginger Peach Tea 🍑"],
        'workout': ["Dance Break!"],
        'podcast': ["How I Built This"],
        'affirmation': ["Embrace the unexpected — it may be a gift in disguise."]
    },
    'neutral': {
        'music': ["'Balanced Beats' playlist"],
        'breathing': ["Simple Belly Breathing"],
        'tea': ["Green Tea 🍵"],
        'workout': ["Gentle Full Body Stretch"],
        'podcast': ["Daily Wellness Snippet"],
        'affirmation': ["Stay grounded. Your calm is your strength."]
    }
}


def get_db_connection():
    conn = sqlite3.connect('health_bright.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create personal_info table (Fixed syntax and missing columns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS personal_info (
            user_id INTEGER PRIMARY KEY,
            gender TEXT DEFAULT 'Other',
            age INTEGER DEFAULT 0,
            weight REAL DEFAULT 0,
            height REAL DEFAULT 0,
            chronic_diseases TEXT DEFAULT 'No Chronic Disease',
            vegetarian BOOLEAN DEFAULT 1,
            allergies TEXT DEFAULT 'None',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create contact details table (Name stored here)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contact_details (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')


    # Daily progress tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_progress (
            user_id INTEGER,
            date TEXT,
            calories INTEGER DEFAULT 0,
            water REAL DEFAULT 0,
            steps INTEGER DEFAULT 0,
            sleep REAL DEFAULT 0,
            heart_rate INTEGER DEFAULT NULL,
            blood_pressure TEXT DEFAULT NULL,
            mood TEXT DEFAULT NULL,
            PRIMARY KEY (user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Create exercise table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exercise_name TEXT,
            type TEXT
        )
    ''')	


    # Create diet table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            recommended_diet TEXT
        )
    ''')

#notification settings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            notifications_enabled BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

   # Create diet table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            email TEXT PRIMARY KEY,
            token TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Run the function
initialize_db()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '').strip()
    gender = request.form.get('gender', 'Other').strip()
    age = request.form.get('age', 0)
    weight = request.form.get('weight', 0)
    height = request.form.get('height', 0)
    chronic_disease = request.form.get('chronic_disease', 'No Chronic Disease').strip()
    allergies = request.form.get('allergies', 'None').strip()
    mobile = request.form.get('mobile', 'Not Provided').strip()

    if not username or not email or not password:
        flash("All fields are required!", "danger")
        return redirect(url_for('index'))

    hashed_password = generate_password_hash(password)

    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()
    
    try:
        # Insert into `users` table
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                       (username, email, hashed_password))
        user_id = cursor.lastrowid  # Get new user ID

        # Insert into `personal_info`
        cursor.execute('''
            INSERT INTO personal_info (user_id, gender, age, weight, height, chronic_diseases, vegetarian, allergies)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, gender, age, weight, height, chronic_disease, 1, allergies))

        # Insert into `contact_details` (storing name separately)
        cursor.execute('''
            INSERT INTO contact_details (user_id, name, mobile) VALUES (?, ?, ?)
        ''', (user_id, username, mobile))

        conn.commit()

        # ✅ Automatically log in after signup
        session['user_id'] = user_id
        session['username'] = username

        flash("Signup successful!", "success")
        return redirect(url_for('dashboard'))  # Redirect to dashboard after signup

    except sqlite3.IntegrityError:
        flash("Email already exists. Please log in.", "danger")
        return redirect(url_for('index'))
    finally:
        conn.close()



@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email').strip().lower()
    password = request.form.get('password').strip()

    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[2], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        flash("Login successful!", "success")
        return redirect(url_for('dashboard'))  # ✅ Redirect to dashboard after login
    else:
        flash("Invalid email or password.", "danger")
        return redirect(url_for('index'))



@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "danger")
        return redirect(url_for('index'))
    
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/predict_by_symptoms', methods=['GET', 'POST'])
def predict_by_symptoms():
    all_symptoms = set()
    for symptom_list in dandss['Symptoms']:
        symptoms = symptom_list.lower().strip().replace(' ', '').split(',')
        all_symptoms.update(symptoms)

    all_symptoms_string = ", ".join(sorted(all_symptoms))

    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip().lower().replace(' ', '').split(',')
        symptoms = [symptom for symptom in symptoms if symptom]

        if len(symptoms) < 3:
            return render_template('predict_by_symptoms.html', error="Please enter at least 3 symptoms.", all_symptoms=all_symptoms_string)

        try:
            result = predict_with_threshold(symptoms)
            return render_template('predict_by_symptoms.html',
                                   prediction=result['disease'],
                                   suggested_diet=result['diet'],
                                   suggested_precaution=result['precaution'],
                                   suggested_exercise=result['exercise'],
                                   exercise_precaution=result['exercise_precaution'],
                                   all_symptoms=all_symptoms_string)
        except Exception as e:
            return render_template('predict_by_symptoms.html', error=f"Error: {e}", all_symptoms=all_symptoms_string)

    return render_template('predict_by_symptoms.html', all_symptoms=all_symptoms_string)


@app.route('/recommend_diet', methods=['GET', 'POST'])
def diet_recommendation():
    user_id = session.get("user_id")  # Ensure user is logged in

    # Fetch chronic disease from personal_info table
    conn = sqlite3.connect("health_bright.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chronic_diseases FROM personal_info WHERE user_id = ?", (user_id,))
    chronic_disease = cursor.fetchone()
    conn.close()

    # Default value if no chronic disease is found
    chronic_disease = chronic_disease[0] if chronic_disease and chronic_disease[0] else "No Chronic Disease"

    if request.method == 'POST':
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        allergy = request.form['allergy'].strip()
        preference = request.form['preference'].strip()

        # Predict diet using the existing function
        diet_plan = predict_diet(height, weight, allergy, preference)

        return render_template("recommend_diet.html", 
                               diet_plan=diet_plan.split(" | "), 
                               chronic_disease=chronic_disease)

    return render_template("recommend_diet.html", chronic_disease=chronic_disease)




@app.route('/recommend_exercise', methods=['GET', 'POST'])
def recommend_exercise():
    df = pd.read_csv("full_exercise_recommendation.csv")

    # Standardize column names
    df.rename(columns={"Duration (min)": "Duration"}, inplace=True)
    df["Exercise_Type"] = df["Exercise_Type"].fillna("General Exercise")

    # Fetch unique exercise types
    exercise_types = sorted(df["Exercise_Type"].unique().tolist())

    recommended_exercises = []

    if request.method == 'POST':
        exercise_type = request.form.get("goal")
        weight_loss_goal = request.form.get("weight_loss_goal")

        # Filter dataset based on user-selected exercise type
        filtered_exercises = df[df["Exercise_Type"] == exercise_type]

        if not filtered_exercises.empty:
            # Select top 10-12 random exercises
            selected_exercises = filtered_exercises.sample(n=min(12, len(filtered_exercises)))

            # Format exercises for display
            recommended_exercises = [
                {
                    "name": row["Exercise_Name"],
                    "duration": row["Duration"],
                    "intensity": row["Intensity"],
                    "steps": row["Steps"],
                   "link": row.get("Video_Link", "#") if pd.notna(row.get("Video_Link", "#")) else "#"
                }
                for _, row in selected_exercises.iterrows()
            ]

    return render_template(
        "recommend_exercise.html",
        exercise_types=exercise_types,
        recommended_exercises=recommended_exercises
    )


@app.route('/mood')
def mood():
    return render_template("mood.html")

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json()
    user_input = data.get('mood', '')
    user_id = session['user_id']
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        # Emotion detection (Hugging Face or other)
        response = requests.post(
            "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": user_input}
        )
        result = response.json()
        primary_emotion = sorted(result[0], key=lambda x: x['score'], reverse=True)[0]['label'].lower()
    except Exception as e:
        print("API error:", e)
        primary_emotion = 'neutral'

    rec = emotion_recommendations.get(primary_emotion, emotion_recommendations['neutral'])

    # ✅ Save mood to daily_progress table
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO daily_progress (user_id, date, mood)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, date) DO UPDATE SET mood = excluded.mood
    ''', (user_id, today, primary_emotion))
    conn.commit()
    conn.close()

    return jsonify({
        "emotion": primary_emotion.capitalize(),
        "music": random.choice(rec["music"]),
        "breathing": random.choice(rec["breathing"]),
        "tea": random.choice(rec["tea"]),
        "workout": random.choice(rec["workout"]),
        "podcast": random.choice(rec["podcast"]),
        "affirmation": random.choice(rec["affirmation"])
    })


@app.route('/food_list')
def food_list():
    return render_template('recipes.html')

@app.route('/get_recipe', methods=['GET'])
def get_recipe():
    food_item = request.args.get('food_item')
    if not food_item:
        return {"error": "No food item provided"}, 400

    try:
        details = get_recipe_details(food_item)
        return {
    		"recipe": details['Recipe'],
    		"link": details['Link'],
    		"image": details['Image'],
    		"nutrition": {
        		"protein": float(details['Protein']),
        		"fat": float(details['Fat']),
        		"carbs": float(details['Carbs'])
            }
        }
    except Exception as e:
        return {"error": str(e)}, 500



@app.route('/food_options')
def food_options():
    if 'user_id' not in session:
        flash("Please log in to access this page.", "danger")
        return redirect(url_for('index'))
    return render_template('food_options.html')


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash("Please log in to access your profile.", "danger")
        return redirect(url_for('dashboard'))
    return render_template('profile.html')


@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        flash("Please log in to update your profile.", "danger")
        return redirect(url_for('login'))

    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        gender = request.form.get('gender', 'Other').strip()
        age = request.form.get('age', '').strip()
        weight = request.form.get('weight', '').strip()
        height = request.form.get('height', '').strip()
        chronic_disease = request.form.get('chronic_disease', 'No Chronic Disease').strip()
        allergy = request.form.get('allergy', 'None').strip()

        # Ensure `contact_details` has a record for the user before updating
        cursor.execute("SELECT COUNT(*) FROM contact_details WHERE user_id = ?", (session['user_id'],))
        contact_exists = cursor.fetchone()[0]

        if contact_exists:
            cursor.execute('UPDATE contact_details SET name = ? WHERE user_id = ?', (name, session['user_id']))
        else:
            cursor.execute('INSERT INTO contact_details (user_id, name, mobile) VALUES (?, ?, ?)',
                           (session['user_id'], name, 'Not Provided'))

        # Update or insert into `personal_info`
        cursor.execute('''
            INSERT INTO personal_info (user_id, gender, age, weight, height, chronic_diseases, allergies)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET 
                gender = excluded.gender, 
                age = excluded.age, 
                weight = excluded.weight, 
                height = excluded.height, 
                chronic_diseases = excluded.chronic_diseases, 
                allergies = excluded.allergies
        ''', (session['user_id'], gender, age, weight, height, chronic_disease, allergy))

        conn.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    # Fetch Name from `contact_details`
    cursor.execute('SELECT name FROM contact_details WHERE user_id = ?', (session['user_id'],))
    contact_data = cursor.fetchone()
    name = contact_data[0] if contact_data and contact_data[0] else "Not Provided"

    # Fetch Other Fields from `personal_info`
    cursor.execute('''
        SELECT gender, age, weight, height, chronic_diseases, allergies
        FROM personal_info WHERE user_id = ?
    ''', (session['user_id'],))
    profile_data = cursor.fetchone()

    profile = {
        'name': name,
        'gender': profile_data[0] if profile_data else 'Other',
        'age': profile_data[1] if profile_data else '',
        'weight': profile_data[2] if profile_data else '',
        'height': profile_data[3] if profile_data else '',
        'chronic_disease': profile_data[4] if profile_data else 'No Chronic Disease',
        'allergy': profile_data[5] if profile_data else 'None',
    }

    conn.close()
    return render_template('update_profile.html', profile=profile)



@app.route('/profile/notifications', methods=['GET', 'POST'])
def notifications():
    if 'user_id' not in session:
        return "Unauthorized", 401

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        notifications_enabled = 1 if request.form.get('notifications_enabled') == 'on' else 0
        cursor.execute("""
            INSERT INTO user_settings (user_id, notifications_enabled) 
            VALUES (?, ?) 
            ON CONFLICT(user_id) 
            DO UPDATE SET notifications_enabled = ?
        """, (user_id, notifications_enabled, notifications_enabled))
        conn.commit()
        conn.close()

        flash("Notification settings updated successfully!", "success")
        return redirect(url_for('profile'))  # Redirect to profile page

    cursor.execute("SELECT notifications_enabled FROM user_settings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    enabled = bool(row['notifications_enabled']) if row else False
    return render_template('notification.html', enabled=enabled)


@app.route('/get_notification_setting', methods=['GET'])
def get_notification_setting():
    if 'user_id' not in session:
        return jsonify({"enabled": False})
    
    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT notifications_enabled FROM user_settings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    enabled = bool(row['notifications_enabled']) if row else False
    return jsonify({"enabled": enabled})



@app.route('/profile/help', methods=['GET', 'POST'])
def help_page():
    if request.method == 'POST':
        flash('Help request submitted!', 'success')
    return render_template('help.html')




@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        user_email = request.form['email'].strip().lower()
        reset_token = secrets.token_urlsafe(16)

        conn = sqlite3.connect('health_bright.db')
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO password_resets (email, token) VALUES (?, ?)", 
                       (user_email, reset_token))
        conn.commit()
        conn.close()

        reset_link = url_for('confirm_reset', token=reset_token, _external=True)

        try:
            msg = Message(
                subject="Password Reset for Health Bright",
                recipients=[user_email],
                body=f"Click here to reset your password:\n{reset_link}\nIf you didn't request this, ignore this email."
            )
            mail.send(msg)
            flash("Password reset link sent! Check your email.", "info")
        except Exception as e:
            flash(f"Failed to send email: {e}", "danger")

        return redirect(url_for('reset_password'))

    return render_template('reset_password.html')


@app.route('/track_progress', methods=['GET', 'POST'])
def track_progress():
    if 'user_id' not in session:
        flash("Please log in to access progress tracking.", "danger")
        return redirect(url_for('index'))

    user_id = session['user_id']
    today = datetime.now().strftime('%Y-%m-%d')
    message = None

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        # Get basic required values
        calories = int(request.form['calories'])
        water = float(request.form['water'])
        steps = int(request.form['steps'])
        sleep = float(request.form['sleep'])

        # Get optional inputs
        heart_rate = request.form.get('heart_rate')
        blood_pressure = request.form.get('blood_pressure')
        mood = request.form.get('mood', '')

        # If heart rate is missing, get last saved
        if not heart_rate or heart_rate.strip() == "":
            cursor.execute('''
                SELECT heart_rate FROM daily_progress
                WHERE user_id = ? AND heart_rate IS NOT NULL
                ORDER BY date DESC LIMIT 1
            ''', (user_id,))
            last_hr = cursor.fetchone()
            heart_rate = int(last_hr['heart_rate']) if last_hr else None
        else:
            heart_rate = int(heart_rate)

        # If blood pressure is missing, get last saved
        if not blood_pressure or blood_pressure.strip() == "":
            cursor.execute('''
                SELECT blood_pressure FROM daily_progress
                WHERE user_id = ? AND blood_pressure IS NOT NULL
                ORDER BY date DESC LIMIT 1
            ''', (user_id,))
            last_bp = cursor.fetchone()
            blood_pressure = last_bp['blood_pressure'] if last_bp else None

        # Save to database (insert or update)
        cursor.execute('''
            INSERT INTO daily_progress (user_id, date, calories, water, steps, sleep, heart_rate, blood_pressure, mood)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET 
                calories = excluded.calories,
                water = excluded.water,
                steps = excluded.steps,
                sleep = excluded.sleep,
                heart_rate = excluded.heart_rate,
                blood_pressure = excluded.blood_pressure,
                mood = excluded.mood
        ''', (user_id, today, calories, water, steps, sleep, heart_rate, blood_pressure, mood))
        conn.commit()

        # 30+ motivational & diagnostic conditions
        if calories >= 500 and calories <= 2500 and steps >= 10000 and water >= 3 and sleep >= 8:
            message = "🌟 You’re the health legend today! Balanced fuel, motion, hydration, and rest—pure magic in motion."
        elif calories >= 400 and steps >= 8000 and water >= 2.5 and sleep >= 7:
            message = "💪 Solid day! You’re carving out success with every step and sip."
        elif calories >= 300 and steps >= 6000 and water >= 2 and sleep >= 6:
            message = "👌 Not perfect, but progress. Keep the momentum, the climb is worth it."
        elif calories < 500 and steps < 5000 and water < 1.5 and sleep < 6:
            message = "⚠️ Warning: Your body’s on empty. Time to refuel, move, hydrate, and rest before it protests louder."
        elif calories > 2800:
            message = "🔥 Easy there! Don’t let overfuel slow your vibe—quality over quantity, always."
        elif steps < 3000:
            message = "🦥 Wake up those legs! Even a stroll counts. The couch isn’t winning today."
        elif water < 1:
            message = "💧 Your body’s deserting you—hydrate like your glow depends on it (because it does)."
        elif sleep < 5:
            message = "😴 Sleep crisis! Your mind and muscles need the ultimate recharge, pronto."
        elif calories >= 2000 and steps < 4000:
            message = "🍔 Too much fuel, too little burn. Balance is the secret sauce here."
        elif water >= 4 and sleep >= 9:
            message = "💧🛌 Hydration + rest = your secret power combo. Keep riding this wave."
        elif steps >= 12000 and calories < 1800:
            message = "🏃‍♂️ Burning up the track but don’t forget to refuel—fuel’s the firestarter."
        elif calories < 800 and sleep < 6:
            message = "🥄 Low fuel and low rest? Your body’s sending SOS signals—listen up."
        elif water < 2 and steps >= 8000:
            message = "🚶‍♂️ Great movement but don’t forget your water bottle—it’s your best sidekick."
        elif calories >= 1500 and water < 1.5:
            message = "🍽️ Solid meals, weak hydration—balance the liquid love for max power."
        elif sleep >= 8 and steps < 4000:
            message = "🛌 Rest master! But movement’s missing—your body’s craving a dance."
        elif calories >= 500 and steps < 6000 and water >= 2 and sleep >= 7:
            message = "🥉 Almost there! Step it up a notch to match your great hydration and rest."
        elif calories < 1000 and steps >= 7000 and water < 1:
            message = "🏃‍♀️ Burning bright but barely sipping—hydrate or face the crash."
        elif sleep < 7 and calories >= 1800:
            message = "😴 High fuel, low sleep—your engine’s running hot. Time to cool down with rest."
        elif steps >= 9000 and water >= 3 and sleep < 6:
            message = "🚀 Stellar steps and hydration but lack of sleep is a speed bump. Catch those Zzz’s!"
        elif calories < 600 and steps < 4000 and water < 1.5 and sleep < 6:
            message = "⚡️ Low energy alert! Your body’s whispering for fuel, movement, and rest. Time to listen."
        elif calories >= 2200 and steps >= 11000 and water >= 3 and sleep >= 7:
            message = "🔥 Firestarter day! You’re fueling and moving like a champ, with hydration on point."
        elif calories < 1500 and steps >= 9000 and water >= 2 and sleep >= 8:
            message = "🌿 Light on calories but heavy on hustle and rest. Balance that plate and keep thriving."
        elif calories > 2800 and steps < 5000 and water < 2:
            message = "⚖️ Fuel overload, movement drought, and dehydration risk—time to rebalance the scales."
        elif calories >= 1800 and steps >= 7000 and water < 1.5 and sleep < 7:
            message = "🔥 Enough fuel, solid steps, but hydration and rest need a serious upgrade."
        elif calories < 800 and steps < 3000 and water >= 2.5 and sleep >= 7:
            message = "🍃 Low fuel and movement, but hydration and rest got you covered. Let’s build from here."
        elif calories >= 2000 and steps >= 12000 and water < 2 and sleep >= 8:
            message = "🏃‍♀️ Fuel and steps are a power duo, but your hydration could use some love."
        elif calories >= 500 and steps < 6000 and water < 1 and sleep < 6:
            message = "💤 Your body’s sending urgent signals: more moves, more water, more sleep!"
        elif calories >= 1000 and steps >= 5000 and water >= 3 and sleep < 5:
            message = "⏳ Fuel and hydration good, but you’re running on empty with sleep. Fix that ASAP."
        elif calories >= 2500 and steps < 7000 and water >= 2 and sleep >= 6:
            message = "🍽️ Plenty of fuel but movement is lagging. Time to shake things up!"
        elif calories < 1200 and steps >= 8000 and water < 1.5 and sleep >= 7:
            message = "🚶‍♂️ Good steps and rest but your body’s hungry and thirsty—don’t ghost your needs."
        elif calories >= 1500 and steps >= 9000 and water >= 3 and sleep < 7:
            message = "🔥 Almost perfect day but your body craves more shut-eye to seal the deal."
        elif calories < 700 and steps < 2000 and water < 1 and sleep < 5:
            message = "⚠️ Emergency mode! Your body’s depleted. Time for a full reset."
        elif calories >= 3000 and steps >= 10000 and water >= 4 and sleep >= 8:
            message = "🎉 Absolute powerhouse! You’re crushing calories, steps, hydration, and rest."
        elif calories < 900 and steps >= 7000 and water >= 2 and sleep < 6:
            message = "🍃 Low fuel and sleep—but your movement and hydration are shining bright."
        elif calories >= 1800 and steps < 4000 and water < 1 and sleep >= 8:
            message = "🍽️ Calories up, but you need to get those steps and water numbers climbing."
        elif calories >= 2200 and steps >= 8000 and water < 1.5 and sleep >= 6:
            message = "🔥 Great fuel and steps, but your hydration game is slipping—sip more!"
        elif calories < 1300 and steps < 5000 and water >= 2 and sleep >= 7:
            message = "🍂 Low fuel and steps but solid hydration and rest. Let’s build momentum."
        elif calories >= 1600 and steps >= 10000 and water >= 3 and sleep >= 9:
            message = "✨ Peak performance! You’re balancing all pillars like a health ninja."
        else:
            message = "🌱 Every day is a new chapter—keep writing your story with care and courage."

    # 📊 Monthly summary
    cursor.execute('''
        SELECT 
            strftime('%Y-%m', date) AS month,
            ROUND(AVG(calories),1),
            ROUND(AVG(water),1),
            ROUND(AVG(steps),1),
            ROUND(AVG(sleep),1),
            ROUND(AVG(heart_rate),1)
        FROM daily_progress
        WHERE user_id = ?
        GROUP BY month
        ORDER BY month DESC
    ''', (user_id,))
    monthly_data = cursor.fetchall()
    conn.close()

    return render_template('track_progress.html', message=message, monthly_data=monthly_data)
@app.route('/update_contact', methods=['GET', 'POST'])
def update_contact():
    if 'user_id' not in session:
        flash("Please log in to update your contact details.", "danger")
        return redirect(url_for('login'))

    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']

        # Update email in users table
        cursor.execute('''
            UPDATE users SET email = ? WHERE id = ?
        ''', (email, session['user_id']))

        # Update name and mobile in contact_details table
        cursor.execute('''
            INSERT INTO contact_details (user_id, name, mobile) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET name = excluded.name, mobile = excluded.mobile;
        ''', (session['user_id'], name, mobile))

        conn.commit()
        conn.close()

        flash("Contact details updated successfully!", "success")
        return redirect(url_for('profile'))

    # Fetch email from users table
    cursor.execute('''
        SELECT email FROM users WHERE id = ?
    ''', (session['user_id'],))
    user_data = cursor.fetchone()

    # Fetch name and mobile from contact_details table
    cursor.execute('''
        SELECT name, mobile FROM contact_details WHERE user_id = ?
    ''', (session['user_id'],))
    contact_data = cursor.fetchone() or ("Not Found", "Not Found")  # Ensure contact_data is always initialized

    # Assign correct values
    contact = {
        "name": contact_data[0],  # Name from contact_details
        "mobile": contact_data[1],  # Mobile from contact_details
        "email": user_data[0] if user_data else 'Not Found'  # Email from users
    }

    conn.close()
    return render_template('update_contact.html', contact=contact)


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        flash("Please log in to change your password.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        old_password = request.form['old_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        conn = sqlite3.connect('health_bright.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE id = ?", (session['user_id'],))
        row = cursor.fetchone()
        conn.close()

        if not row:
            flash("User not found.", "danger")
            return redirect(url_for('change_password'))

        stored_password = row[0]

        # Check old password with hashing
        if not check_password_hash(stored_password, old_password):
            flash("Old password is incorrect!", "danger")
            return redirect(url_for('change_password'))

        # Validate new passwords
        if new_password != confirm_password:
            flash("New passwords do not match!", "danger")
            return redirect(url_for('change_password'))

        # Hash the new password before storing it
        hashed_password = generate_password_hash(new_password)

        # Update the password in the database
        conn = sqlite3.connect('health_bright.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, session['user_id']))
        conn.commit()
        conn.close()

        flash("Password updated successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('change_password.html')

@app.route("/build_meal", methods=["GET", "POST"])
def build_meal():
    if request.method == "POST":
        ingredients = request.form.get("ingredients")
        if ingredients:
            try:
                # First API call to get basic recipes
                response = requests.get(SPOONACULAR_ENDPOINT, params={
                    "ingredients": ingredients,
                    "number": 5,
                    "ranking": 2,
                    "ignorePantry": True,
                    "apiKey": API_KEY
                })
                data = response.json()

                recipes = []
                for item in data:
                    recipe_id = item["id"]
                    # Second API call for detailed instructions
                    details_response = requests.get(
                        f"https://api.spoonacular.com/recipes/{recipe_id}/information",
                        params={"apiKey": API_KEY}
                    )
                    details = details_response.json()
                    recipes.append({
                        "title": item["title"],
                        "image": item["image"],
                        "instructions": details.get("instructions", "No instructions available."),
                    })

                return render_template("build_meal.html", recipes=recipes)

            except Exception as e:
                return render_template("build_meal.html", error="API error occurred. Please retry.")
        else:
            return render_template("build_meal.html", error="Please enter ingredients.")
    return render_template("build_meal.html")


@app.route('/reset/<token>', methods=['GET', 'POST'])
def confirm_reset(token):
    conn = sqlite3.connect('health_bright.db')
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM password_resets WHERE token = ?", (token,))
    row = cursor.fetchone()

    if not row:
        flash("Invalid or expired reset link.", "danger")
        return redirect(url_for('index'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        hashed_password = generate_password_hash(new_password)

        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, row[0]))
        cursor.execute("DELETE FROM password_resets WHERE email = ?", (row[0],))
        conn.commit()
        conn.close()

        flash("Password updated successfully! Please log in.", "success")
        return redirect(url_for('index'))

    return render_template('confirm_reset.html', token=token)

if __name__ == '__main__':
    app.run(debug=True)
