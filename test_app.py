import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# A very simple pipeline with only standard sklearn components
# NO SMOTE, NO ColumnTransformer for this test.
simple_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Use only the numerical features from your training data for this test
simple_features = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 
                 'S2_Light', 'S3_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 
                 'S5_CO2', 'S5_CO2_Slope']

# Train this simple pipeline
simple_pipeline.fit(X_train[simple_features], y_train)

# Save this simple pipeline to a NEW file
joblib.dump(simple_pipeline, 'simple_model.joblib')

print("Simple test model saved to simple_model.joblib")