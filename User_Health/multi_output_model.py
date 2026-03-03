"""
Multi-Output Regression Model for Daily Nutrient Recommendations
Predicts: calories, protein, carbs, fat, sugar_limit, sodium_limit based on user profile
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

# ============ STEP 1: CREATE TRAINING DATA ============

def calculate_daily_targets(age, gender, weight_kg, height_cm, activity_level, 
                           has_diabetes, has_hypertension, has_obesity, 
                           has_kidney_disease, health_goal):
    """
    Calculate recommended daily nutrient targets based on user profile
    This is the logic you'll encode in training data
    """
    
    # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
    if gender == 'Male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    # Activity multiplier
    activity_multipliers = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    
    activity_mult = activity_multipliers.get(activity_level, 1.55)
    
    # Base calorie target
    calories = bmr * activity_mult
    
    # Adjust based on health goal
    if health_goal == 'Weight Loss':
        calories *= 0.8  # 20% deficit
    elif health_goal == 'Weight Gain':
        calories *= 1.15  # 15% surplus
    elif health_goal == 'Muscle Building':
        calories *= 1.1  # 10% surplus
    
    # Adjust for obesity
    if has_obesity:
        calories = min(calories, 1800)  # Cap for weight loss
    
    # Protein calculation (g per kg body weight)
    if health_goal == 'Muscle Building':
        protein = weight_kg * 2.0  # High protein
    elif health_goal == 'Weight Loss':
        protein = weight_kg * 1.6  # Preserve muscle
    else:
        protein = weight_kg * 1.2  # Maintenance
    
    # Adjust for kidney disease (restrict protein)
    if has_kidney_disease:
        protein = min(protein, weight_kg * 0.8)  # Low protein
    
    # Carbohydrates (45-65% of calories)
    if has_diabetes:
        carbs = (calories * 0.40) / 4  # 40% of calories, lower for diabetics
    else:
        carbs = (calories * 0.50) / 4  # 50% of calories
    
    # Fat (20-35% of calories)
    fat = (calories * 0.30) / 9  # 30% of calories
    
    # Sugar limit (WHO recommendation: <10% of calories)
    if has_diabetes:
        sugar_limit = min(30, (calories * 0.05) / 4)  # 5% for diabetics, max 30g
    else:
        sugar_limit = (calories * 0.10) / 4  # 10% for healthy people
    
    # Sodium limit
    if has_hypertension:
        sodium_limit = 1500  # Strict limit for high BP
    elif has_kidney_disease:
        sodium_limit = 1500  # Strict for kidney issues
    else:
        sodium_limit = 2300  # Normal limit
    
    # Fiber recommendation
    fiber = 25 if gender == 'Female' else 38
    
    return {
        'daily_calories_target': round(calories, 0),
        'daily_protein_target': round(protein, 1),
        'daily_carbs_target': round(carbs, 1),
        'daily_fat_target': round(fat, 1),
        'daily_sugar_limit': round(sugar_limit, 1),
        'daily_sodium_limit': round(sodium_limit, 0),
        'daily_fiber_target': fiber
    }


def generate_training_data(num_samples=5000):
    """Generate training data with user profiles and their recommended daily targets"""
    
    np.random.seed(42)
    data = []
    
    for i in range(num_samples):
        # User profile
        age = np.random.randint(18, 80)
        gender = np.random.choice(['Male', 'Female'])
        weight_kg = np.random.uniform(45, 120)
        height_cm = np.random.uniform(150, 195)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        activity_level = np.random.choice([
            'Sedentary', 'Lightly Active', 'Moderately Active', 
            'Very Active', 'Extremely Active'
        ], p=[0.3, 0.25, 0.25, 0.15, 0.05])
        
        # Health conditions
        has_diabetes = np.random.choice([0, 1], p=[0.75, 0.25])
        has_hypertension = np.random.choice([0, 1], p=[0.7, 0.3])
        has_thyroid = np.random.choice([0, 1], p=[0.85, 0.15])
        has_kidney_disease = np.random.choice([0, 1], p=[0.92, 0.08])
        has_obesity = 1 if bmi > 30 else 0
        
        # Health goal
        health_goal = np.random.choice([
            'Weight Loss', 'Weight Gain', 'Muscle Building', 
            'Maintain Health', 'Manage Disease'
        ], p=[0.35, 0.08, 0.12, 0.35, 0.1])
        
        # Calculate targets using the formula
        targets = calculate_daily_targets(
            age, gender, weight_kg, height_cm, activity_level,
            has_diabetes, has_hypertension, has_obesity,
            has_kidney_disease, health_goal
        )
        
        # Store sample
        data.append({
            # Input features
            'age': age,
            'gender': gender,
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'bmi': bmi,
            'activity_level': activity_level,
            'has_diabetes': has_diabetes,
            'has_hypertension': has_hypertension,
            'has_thyroid': has_thyroid,
            'has_kidney_disease': has_kidney_disease,
            'has_obesity': has_obesity,
            'health_goal': health_goal,
            
            # Target outputs (what model will predict)
            'daily_calories_target': targets['daily_calories_target'],
            'daily_protein_target': targets['daily_protein_target'],
            'daily_carbs_target': targets['daily_carbs_target'],
            'daily_fat_target': targets['daily_fat_target'],
            'daily_sugar_limit': targets['daily_sugar_limit'],
            'daily_sodium_limit': targets['daily_sodium_limit'],
            'daily_fiber_target': targets['daily_fiber_target']
        })
    
    return pd.DataFrame(data)


# ============ STEP 2: TRAIN MULTI-OUTPUT MODEL ============

def train_multi_output_model():
    """Train a model that predicts 7 nutrient targets simultaneously"""
    
    print("🔄 Generating training data...")
    df = generate_training_data(num_samples=10000)
    
    # Save sample
    df.head(200).to_csv('/mnt/user-data/outputs/multi_output_training_data.csv', index=False)
    print(f"✅ Generated {len(df)} samples")
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_activity = LabelEncoder()
    le_goal = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['activity_encoded'] = le_activity.fit_transform(df['activity_level'])
    df['goal_encoded'] = le_goal.fit_transform(df['health_goal'])
    
    # Feature columns (input)
    feature_columns = [
        'age', 'gender_encoded', 'weight_kg', 'height_cm', 'bmi',
        'activity_encoded', 'has_diabetes', 'has_hypertension',
        'has_thyroid', 'has_kidney_disease', 'has_obesity', 'goal_encoded'
    ]
    
    # Target columns (output) - MULTIPLE OUTPUTS!
    target_columns = [
        'daily_calories_target',
        'daily_protein_target',
        'daily_carbs_target',
        'daily_fat_target',
        'daily_sugar_limit',
        'daily_sodium_limit',
        'daily_fiber_target'
    ]
    
    X = df[feature_columns]
    y = df[target_columns]  # Multiple outputs!
    
    print(f"\n📊 Model Configuration:")
    print(f"   Input Features: {len(feature_columns)}")
    print(f"   Output Targets: {len(target_columns)}")
    print(f"   Outputs: {', '.join(target_columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Multi-Output Random Forest
    print("\n🧠 Training Multi-Output Random Forest...")
    print("   (This predicts ALL 7 nutrients simultaneously!)")
    
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # MultiOutputRegressor wraps the model to handle multiple outputs
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate each output
    print(f"\n📈 Model Performance (per output):")
    print("=" * 80)
    
    for i, target in enumerate(target_columns):
        train_mae = mean_absolute_error(y_train[target], y_train_pred[:, i])
        test_mae = mean_absolute_error(y_test[target], y_test_pred[:, i])
        train_r2 = r2_score(y_train[target], y_train_pred[:, i])
        test_r2 = r2_score(y_test[target], y_test_pred[:, i])
        
        print(f"\n{target}:")
        print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
        print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    
    # Overall performance
    overall_train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
    overall_test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
    
    print("\n" + "=" * 80)
    print(f"📊 Overall Model R² Score:")
    print(f"   Training: {overall_train_r2:.4f}")
    print(f"   Testing: {overall_test_r2:.4f}")
    
    # Save model and preprocessing objects
    with open('/mnt/user-data/outputs/multi_output_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('/mnt/user-data/outputs/multi_output_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label encoders
    encoders = {
        'gender': le_gender,
        'activity_level': le_activity,
        'health_goal': le_goal
    }
    with open('/mnt/user-data/outputs/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save feature and target column names
    with open('/mnt/user-data/outputs/column_names.pkl', 'wb') as f:
        pickle.dump({
            'features': feature_columns,
            'targets': target_columns
        }, f)
    
    print("\n✅ Model saved successfully!")
    
    return model, scaler, encoders, feature_columns, target_columns


# ============ STEP 3: PREDICTION FUNCTION ============

def predict_daily_targets(user_profile: dict, 
                         model_path='/mnt/user-data/outputs/multi_output_model.pkl'):
    """
    Predict ALL daily nutrient targets for a user in ONE prediction
    
    Args:
        user_profile: dict with keys:
            - age, gender, weight_kg, height_cm, activity_level
            - has_diabetes, has_hypertension, has_thyroid, 
              has_kidney_disease, health_goal
    
    Returns:
        dict with all 7 predicted targets
    """
    
    # Load model and preprocessing
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open('/mnt/user-data/outputs/multi_output_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('/mnt/user-data/outputs/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('/mnt/user-data/outputs/column_names.pkl', 'rb') as f:
        columns = pickle.load(f)
    
    # Calculate BMI
    bmi = user_profile['weight_kg'] / ((user_profile['height_cm'] / 100) ** 2)
    has_obesity = 1 if bmi > 30 else 0
    
    # Encode categorical features
    gender_encoded = encoders['gender'].transform([user_profile['gender']])[0]
    activity_encoded = encoders['activity_level'].transform([user_profile['activity_level']])[0]
    goal_encoded = encoders['health_goal'].transform([user_profile['health_goal']])[0]
    
    # Prepare features
    features = pd.DataFrame([{
        'age': user_profile['age'],
        'gender_encoded': gender_encoded,
        'weight_kg': user_profile['weight_kg'],
        'height_cm': user_profile['height_cm'],
        'bmi': bmi,
        'activity_encoded': activity_encoded,
        'has_diabetes': user_profile.get('has_diabetes', 0),
        'has_hypertension': user_profile.get('has_hypertension', 0),
        'has_thyroid': user_profile.get('has_thyroid', 0),
        'has_kidney_disease': user_profile.get('has_kidney_disease', 0),
        'has_obesity': has_obesity,
        'goal_encoded': goal_encoded
    }])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)[0]  # Returns array of 7 values
    
    # Create result dictionary
    result = {}
    for i, target_name in enumerate(columns['targets']):
        result[target_name] = round(predictions[i], 1)
    
    # Add calculated per-meal limits
    result['max_calories_per_meal'] = round(result['daily_calories_target'] * 0.40, 0)
    result['max_protein_per_meal'] = round(result['daily_protein_target'] * 0.40, 1)
    result['max_sugar_per_meal'] = round(result['daily_sugar_limit'] * 0.33, 1)
    result['max_sodium_per_meal'] = round(result['daily_sodium_limit'] * 0.30, 0)
    
    return result


# ============ RUN TRAINING AND TESTING ============

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-OUTPUT REGRESSION MODEL FOR DAILY NUTRIENT TARGETS")
    print("=" * 80)
    
    # Train model
    model, scaler, encoders, features, targets = train_multi_output_model()
    
    # Test predictions
    print("\n" + "=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)
    
    # Test case 1: Diabetic patient wanting weight loss
    print("\n🧪 TEST 1: Diabetic Female, Weight Loss Goal")
    print("-" * 80)
    
    test_user_1 = {
        'age': 45,
        'gender': 'Female',
        'weight_kg': 75,
        'height_cm': 160,
        'activity_level': 'Lightly Active',
        'has_diabetes': 1,
        'has_hypertension': 0,
        'has_thyroid': 0,
        'has_kidney_disease': 0,
        'health_goal': 'Weight Loss'
    }
    
    predictions_1 = predict_daily_targets(test_user_1)
    
    print(f"User Profile: {test_user_1['age']}y Female, {test_user_1['weight_kg']}kg, {test_user_1['height_cm']}cm")
    print(f"Conditions: Diabetic | Goal: Weight Loss")
    print(f"\n📊 Predicted Daily Targets:")
    print(f"   Calories: {predictions_1['daily_calories_target']:.0f} kcal")
    print(f"   Protein: {predictions_1['daily_protein_target']:.1f}g")
    print(f"   Carbs: {predictions_1['daily_carbs_target']:.1f}g")
    print(f"   Fat: {predictions_1['daily_fat_target']:.1f}g")
    print(f"   Sugar Limit: {predictions_1['daily_sugar_limit']:.1f}g (strict for diabetics)")
    print(f"   Sodium Limit: {predictions_1['daily_sodium_limit']:.0f}mg")
    print(f"   Fiber: {predictions_1['daily_fiber_target']:.0f}g")
    print(f"\n🍽️ Per-Meal Limits (40% rule):")
    print(f"   Max Calories/Meal: {predictions_1['max_calories_per_meal']:.0f} kcal")
    print(f"   Max Protein/Meal: {predictions_1['max_protein_per_meal']:.1f}g")
    print(f"   Max Sugar/Meal: {predictions_1['max_sugar_per_meal']:.1f}g")
    print(f"   Max Sodium/Meal: {predictions_1['max_sodium_per_meal']:.0f}mg")
    
    # Test case 2: Athlete building muscle
    print("\n\n🧪 TEST 2: Male Athlete, Muscle Building")
    print("-" * 80)
    
    test_user_2 = {
        'age': 25,
        'gender': 'Male',
        'weight_kg': 80,
        'height_cm': 180,
        'activity_level': 'Very Active',
        'has_diabetes': 0,
        'has_hypertension': 0,
        'has_thyroid': 0,
        'has_kidney_disease': 0,
        'health_goal': 'Muscle Building'
    }
    
    predictions_2 = predict_daily_targets(test_user_2)
    
    print(f"User Profile: {test_user_2['age']}y Male, {test_user_2['weight_kg']}kg, {test_user_2['height_cm']}cm")
    print(f"Conditions: Healthy | Goal: Muscle Building")
    print(f"\n📊 Predicted Daily Targets:")
    print(f"   Calories: {predictions_2['daily_calories_target']:.0f} kcal (high for muscle gain)")
    print(f"   Protein: {predictions_2['daily_protein_target']:.1f}g (2g per kg bodyweight)")
    print(f"   Carbs: {predictions_2['daily_carbs_target']:.1f}g")
    print(f"   Fat: {predictions_2['daily_fat_target']:.1f}g")
    print(f"   Sugar Limit: {predictions_2['daily_sugar_limit']:.1f}g")
    print(f"   Sodium Limit: {predictions_2['daily_sodium_limit']:.0f}mg")
    print(f"   Fiber: {predictions_2['daily_fiber_target']:.0f}g")
    
    # Test case 3: Hypertension patient
    print("\n\n🧪 TEST 3: Hypertension Patient, Maintain Health")
    print("-" * 80)
    
    test_user_3 = {
        'age': 60,
        'gender': 'Male',
        'weight_kg': 85,
        'height_cm': 175,
        'activity_level': 'Moderately Active',
        'has_diabetes': 0,
        'has_hypertension': 1,
        'has_thyroid': 0,
        'has_kidney_disease': 0,
        'health_goal': 'Maintain Health'
    }
    
    predictions_3 = predict_daily_targets(test_user_3)
    
    print(f"User Profile: {test_user_3['age']}y Male, {test_user_3['weight_kg']}kg, {test_user_3['height_cm']}cm")
    print(f"Conditions: Hypertension | Goal: Maintain Health")
    print(f"\n📊 Predicted Daily Targets:")
    print(f"   Calories: {predictions_3['daily_calories_target']:.0f} kcal")
    print(f"   Protein: {predictions_3['daily_protein_target']:.1f}g")
    print(f"   Carbs: {predictions_3['daily_carbs_target']:.1f}g")
    print(f"   Fat: {predictions_3['daily_fat_target']:.1f}g")
    print(f"   Sugar Limit: {predictions_3['daily_sugar_limit']:.1f}g")
    print(f"   Sodium Limit: {predictions_3['daily_sodium_limit']:.0f}mg (STRICT for hypertension!)")
    print(f"   Fiber: {predictions_3['daily_fiber_target']:.0f}g")
    print(f"\n🍽️ Per-Meal Sodium Limit: {predictions_3['max_sodium_per_meal']:.0f}mg (must stay low!)")
    
    print("\n" + "=" * 80)
