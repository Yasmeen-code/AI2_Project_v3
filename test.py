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

# Display the lists of numerical and categorical columns
print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", non_numerical_columns)

# Display unique values for each categorical column
for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {df[col].unique()}")

column_name = 'Plant_Health_Status'
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
ax = sns.countplot(y=column_name, data=df, palette='muted')
plt.title(f'Distribution of {column_name}')
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height()/2),
                ha='center', va='center', xytext=(10, 0), textcoords='offset points')
sns.despine()

plt.subplot(1, 2, 2)
vals = df[column_name].value_counts()
vals.plot.pie(autopct='%1.1f%%', startangle=90,
              explode=[0.05]*vals.nunique(), colors=sns.color_palette('muted'))
plt.title(f'Percentage Distribution of {column_name}')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Find the earliest and latest timestamps
start_date = pd.to_datetime(df['Timestamp']).min()
end_date = pd.to_datetime(df['Timestamp']).max()

print("Start Date:", start_date)
print("End Date:", end_date)

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Check time differences between consecutive entries
time_diffs = df['Timestamp'].diff().value_counts()

print("Most common time differences:")
print(time_diffs.head(10))

# Function to perform univariate analysis for numeric columns
def univariate_analysis(data, columns):
    plt.figure(figsize=(16, 20))  
    
    muted_colors = sns.color_palette("muted", len(columns))
    
    for i, column in enumerate(columns):
        plt.subplot(4, 3, i + 1)  
        sns.histplot(data[column], kde=True, bins=10, color=muted_colors[i])
        plt.title(f'{column.replace("_", " ")} Distribution with KDE')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

columns_to_analyze = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', 
    'Electrochemical_Signal'
]

# Perform univariate analysis
univariate_analysis(df, columns_to_analyze)

# Function to perform univariate analysis for numeric columns (ALL IN ONE PAGE)
def univariate_boxplots(data, columns):
    n = len(columns)
    rows = (n // 3) + 1
    plt.figure(figsize=(18, rows * 2.5)) 
    
    colors = sns.color_palette("muted", n)
    
    for i, column in enumerate(columns):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(x=data[column], color=colors[i])
        plt.title(column.replace("_", " "))
        plt.tight_layout()

    plt.show()
    
    # Show statistics after plotting
    for column in columns:
        print(f"\nSummary Statistics for {column}:\n")
        print(data[column].describe())

# Run all boxplots together
univariate_boxplots(df, columns_to_analyze)


# Convert the Timestamp column to datetime format if not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create a 'Week' column to group by week
df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)

# Aggregate Plant Health Status weekly for each Plant ID
weekly_health_status = (
    df.groupby(['Plant_ID', 'Week', 'Plant_Health_Status'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

weekly_health_status.columns.name = None
weekly_health_status = weekly_health_status.rename(columns={
    'High Stress': 'High_Stress_Count',
    'Moderate Stress': 'Moderate_Stress_Count',
    'Healthy': 'Healthy_Count'
})

weekly_health_status

# Compare Plant_Health_Status with Soil Properties
# Define soil properties
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

# Create subplots to visualize the relationship between Plant_Health_Status and soil properties
plt.figure(figsize=(16, 20))
for i, feature in enumerate(soil_properties):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Environmental Conditions
# Define environmental condition features
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

# Create subplots to visualize the relationship between Plant_Health_Status and environmental conditions
plt.figure(figsize=(16, 12))
for i, feature in enumerate(environmental_conditions):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Plant Health Indicators

# Define plant health indicator features
health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Create subplots to visualize the relationship between Plant_Health_Status and plant health indicators
plt.figure(figsize=(16, 8))
for i, feature in enumerate(health_indicators):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Define nutrient levels
nutrient_levels = ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']

# Group data by Plant_ID and calculate mean nutrient levels
nutrient_mean = df.groupby('Plant_ID')[nutrient_levels].mean()

# Group data by Plant_ID and calculate standard deviation for nutrient levels
nutrient_std = df.groupby('Plant_ID')[nutrient_levels].std()

# Displaying the mean values table
print("------ Mean Nutrient Levels by Plant ID ------")
print(nutrient_mean)

# Displaying the standard deviation table
print("\n------ Standard Deviation of Nutrient Levels by Plant ID ------")
print(nutrient_std)

# Define nutrient levels
nutrients = ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']

# Set up the plot
plt.figure(figsize=(18, 10))

for i, nutrient in enumerate(nutrients):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Plant_ID', y=nutrient, data=df, palette='muted')
    plt.title(f'{nutrient.replace("_", " ")} by Plant ID')
    plt.xlabel('Plant ID')
    plt.ylabel(nutrient.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Visualize the nutrient levels across different Plant_Health_Status categories (High Stress, Moderate Stress, Healthy) for each Plant_ID
# Define nutrient levels
nutrients = ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']
nutrient_status_mean = df.groupby(['Plant_ID', 'Plant_Health_Status'])[
    ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']
].mean().reset_index()

# Set up the plot
plt.figure(figsize=(18, 12))
for i, nutrient in enumerate(nutrients):
    plt.subplot(2, 2, i + 1)
    sns.barplot(
        x='Plant_ID', 
        y=nutrient, 
        hue='Plant_Health_Status', 
        data=nutrient_status_mean, 
        palette='muted'
    )
    plt.title(f'{nutrient.replace("_", " ")} by Plant ID and Health Status')
    plt.xlabel('Plant ID')
    plt.ylabel(nutrient.replace('_', ' '))
    plt.legend(title='Health Status', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#correlational analysis
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

# Count unique values in Plant_Health_Status before encoding
unique_value_counts_before = df['Plant_Health_Status'].value_counts()

# Count unique values in Plant_Health_Status_Encoded after encoding
unique_value_counts_after = df['Plant_Health_Status_Encoded'].value_counts()

# Display unique values before and after encoding
print("----- Unique Values in Plant Health Status Before Encoding ----- ")
print(unique_value_counts_before)

print("\n----- Unique Values in Plant Health Status After Encoding ----- ")
print(unique_value_counts_after)

# Drop the original Plant_Health_Status column
df= df.drop(columns=['Plant_Health_Status', 'Week'])
df= df.drop(columns=['Plant_ID', 'Timestamp'])

# Define features and target
X = df.drop(columns=['Plant_Health_Status_Encoded'])  # Features
y = df['Plant_Health_Status_Encoded']  # Target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shapes of the splits for verification
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Apply StandardScaler
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

# Initialize lists to store results
results = []

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, target_names=['Healthy', 'Moderate Stress', 'High Stress'])
    })

# Display results for each model
for result in results:
    print(f"\nModel: {result['Model']}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    
    # Confusion Matrix Visualization
    cm = result["Confusion Matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Healthy', 'Moderate Stress', 'High Stress'], 
                yticklabels=['Healthy', 'Moderate Stress', 'High Stress'])
    plt.title(f'Confusion Matrix for {result["Model"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print Classification Report
    print(result["Classification Report"])

# Prepare a summary table for model evaluation
evaluation_summary = []

# Extract the key metrics for each model
for result in results:
    evaluation_summary.append({
        "Model": result["Model"],
        "Accuracy": result["Accuracy"],
    })

evaluation_summary_df = pd.DataFrame(evaluation_summary)

# Sort by accuracy and display
print(evaluation_summary_df.sort_values(by="Accuracy", ascending=False))

