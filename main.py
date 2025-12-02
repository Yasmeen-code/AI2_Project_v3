import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_csv('plant_health_data.csv')
print("File Loaded Successfully")
print("Shape of the dataset:", df.shape)
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe().T)
print(f'\nMissing values: {df.isna().sum().sum()}')
print(f'Duplicated values: {df.duplicated().sum()}')
print("\nUnique Values in Each Column:")
print(df.nunique())

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_columns = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", non_numerical_columns)

for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {df[col].unique()}")
#---------------------------------------------------------------------------------
column_name = 'Plant_Health_Status'

# Second subplot: Pie chart
df[column_name].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90, explode=[0.05]*df[column_name].nunique())
plt.title(f'Percentage Distribution of {column_name}')
plt.ylabel('')  

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
# Define columns to analyze
columns_to_analyze = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', 
    'Electrochemical_Signal'
]

def univariate_analysis_all(data, columns):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 20))   
    
    for i, column in enumerate(columns):
        plt.subplot(4, 3, i + 1)  
        
        color = sns.color_palette("muted")[i % len(sns.color_palette("muted"))]
        
        sns.boxplot(x=data[column], color=color)
        
        plt.title(f'{column.replace("_", " ")}', fontsize=12, fontweight="bold")
        plt.xlabel("")  
    
    plt.tight_layout()
    plt.show()
    
    print("\n========== Summary Statistics ==========\n")
    for column in columns:
        print(f"--- {column.replace('_', ' ')} ---")
        print(data[column].describe(), "\n")


univariate_analysis_all(df, columns_to_analyze)
#---------------------------------------------------------------------------------
# Compare Plant_Health_Status with Soil Properties
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

plt.figure(figsize=(16, 20))
for i, feature in enumerate(soil_properties):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df)
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
# Compare Plant_Health_Status with Environmental Conditions
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

plt.figure(figsize=(16, 12))
for i, feature in enumerate(environmental_conditions):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df)
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
# Compare Plant_Health_Status with Plant Health Indicators
health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

plt.figure(figsize=(16, 8))
for i, feature in enumerate(health_indicators):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df)
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
# Calculate mean nutrient levels by Plant ID and Plant Health Status
nutrient_status_mean = df.groupby(['Plant_ID', 'Plant_Health_Status'])[
    ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']
].mean().reset_index()

# Separate the nutrient status mean table into three based on Plant_Health_Status
nutrient_high_stress = nutrient_status_mean[nutrient_status_mean['Plant_Health_Status'] == 'High Stress']
nutrient_moderate_stress = nutrient_status_mean[nutrient_status_mean['Plant_Health_Status'] == 'Moderate Stress']
nutrient_healthy = nutrient_status_mean[nutrient_status_mean['Plant_Health_Status'] == 'Healthy']

# print the three tables to the user
print("----- Nutrient Levels for High Stress Plants -----")
print(nutrient_high_stress)

print("\n----- Nutrient Levels for Moderate Stress Plants -----")
print(nutrient_moderate_stress)

print("\n----- Nutrient Levels for Healthy Plants -----")
print(nutrient_healthy)
#---------------------------------------------------------------------------------
# Correlation heatmap for numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='YlGnBu', 
    square=True, 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Heatmap for Numerical Features', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Custom mapping for Plant_Health_Status
custom_mapping = {'High Stress': 2, 'Moderate Stress': 1, 'Healthy': 0}
df['Plant_Health_Status_Encoded'] = df['Plant_Health_Status'].map(custom_mapping)

drop_columns = ['Plant_ID', 'Plant_Health_Status', 'Timestamp']
df = df.drop(columns=drop_columns)

# Define features and target
X = df.drop(columns=['Plant_Health_Status_Encoded']) 
y = df['Plant_Health_Status_Encoded']  

#  Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply StandardScaler on numeric columns only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "KNN Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("Models initialized:", list(models.keys()))

results = []

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test, y_pred, target_names=['Healthy', 'Moderate Stress', 'High Stress']
        )
    })

#  print results
for result in results:
    print(f"\nModel: {result['Model']}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    # Confusion Matrix Heatmap
    cm = result["Confusion Matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Healthy', 'Moderate Stress', 'High Stress'],
                yticklabels=['Healthy', 'Moderate Stress', 'High Stress'])
    plt.title(f'Confusion Matrix for {result["Model"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Classification report
    print(result["Classification Report"])
