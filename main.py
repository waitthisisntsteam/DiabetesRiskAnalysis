import pandas as pd
import matplotlib.pyplot as plt
# ============================================================================
# ! Data Loading !
# ============================================================================
# Kaggle Datasets:
# > Health & Lifestyle Data for Diabetes Prediction...
df_lifestyle = pd.read_csv('./datasets/Diabetes_and_LifeStyle_Dataset.csv')
# * Link: https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction
# > Diabetes Health Indicators Dataset...
df_indicators = pd.read_csv('./datasets/diabetes_012_health_indicators_BRFSS2015.csv')
# * Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# > Healthcare Diabetes Dataset...
df_healthcare = pd.read_csv('./datasets/Healthcare-Diabetes.csv')
# * Link: https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes
# ============================================================================




# ============================================================================
# ! Key Notes !
# ============================================================================

# - The three datasets have different column names, but measure the same concepts.
# - Data preprocessing is needed to standardize andthe datasets.
# - Focus on key features relevant to diabetes prediction and analysis.
# - Visualizations will compare diabetes prevalence across various factors.

# - Lifestyle and indicators datasets have the most features overlapping.
# - Healthcare dataset has fewer overlapping features, so will be used for general comparisons only.

# =============================================================================




# ============================================================================
# ! Data Preprocessing and Standardization !
# ============================================================================
# Rename columns across datasets for consistency:
df_lifestyle = df_lifestyle.rename(columns={
    'bmi': 'BMI',
    'gender': 'Sex',
    'diastolic_bp': 'DiastolicBP',
    'insulin_level': 'Insulin',
    'cholesterol_total': 'Cholesterol'
})
df_healthcare = df_healthcare.rename(columns={
    'BloodPressure': 'DiastolicBP',
    'Outcome': 'Diabetes_Binary'
})
df_indicators = df_indicators.rename(columns={
    'HeartDiseaseorAttack': 'Heart_Disease_or_Attack',
    'PhysActivity': 'Physically_Active',
    'HvyAlcoholConsump': 'Alcohol_Consumer'
})

# Standardize dataset columns:
# > Age Groups...
age_bins = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
age_labels = ['Under 18', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', 'Over 80']

df_healthcare['Age_Group'] = pd.cut(df_healthcare['Age'], bins=age_bins, labels=age_labels, right=False)
df_lifestyle['Age_Group'] = pd.cut(df_lifestyle['Age'], bins=age_bins, labels=age_labels, right=False)

age_group_map = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49', 
                7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: 'Over 80'}
df_indicators['Age_Group'] = df_indicators['Age'].map(age_group_map)

# > Gluceose levels:
df_lifestyle['Glucose'] = (df_lifestyle['glucose_fasting'] +  df_lifestyle['glucose_postprandial']) / 2

# > Diabetes diagnosis - BINARY ONLY (0 = No, 1 = Yes)
# df_lifestyle: Rename diagnosed_diabetes to Diabetes_Binary
df_lifestyle['Diabetes_Binary'] = df_lifestyle['diagnosed_diabetes']
# df_indicators: Create Diabetes_Binary (1 if Diabetes_012 == 2, else 0)
df_indicators['Diabetes_Binary'] = (df_indicators['Diabetes_012'] == 2).astype(int)

# ! Harmonize categorical variables between **lifestyle** and **indicators** datasets:
# > Income levels...
income_map_indicators = {
    1: 'Low',      # <$10,000
    2: 'Low',      # <$15,000
    3: 'Low',      # <$20,000
    4: 'Low',      # <$25,000
    5: 'Medium',   # <$35,000
    6: 'Medium',   # <$50,000
    7: 'Medium',   # <$75,000
    8: 'High'      # $75,000+
}
df_indicators['Income_Level'] = df_indicators['Income'].map(income_map_indicators)
income_map_lifestyle = {
    'Low': 'Low',
    'Lower-Middle': 'Low',
    'Medium': 'Medium',
    'Upper-Middle': 'Medium',
    'High': 'High'
}
df_lifestyle['Income_Level'] = df_lifestyle['income_level'].map(income_map_lifestyle)


# > Education levels...
education_map_indicators = {
    1: 'No Formal',        # Never attended school or only kindergarten
    2: 'Elementary',       # Grades 1 through 8
    3: 'Some High School', # Grades 9 through 11
    4: 'High School',      # Grade 12 or GED
    5: 'Some College',     # College 1-3 years
    6: 'College Graduate'  # College 4+ years
}
df_indicators['Education_Level'] = df_indicators['Education'].map(education_map_indicators)

education_map_lifestyle = {
    'No formal': 'No Formal',
    'Highschool': 'High School',
    'High school': 'High School',
    'Some College': 'Some College',
    'Bachelor': 'College Graduate',
    'Graduate': 'College Graduate',
    'Masters': 'College Graduate',
    'PhD': 'College Graduate'
}
df_lifestyle['Education_Level'] = df_lifestyle['education_level'].map(education_map_lifestyle)

# > Smoking status...
smoking_map_lifestyle = {
    'Never': 0,    # Never smoked
    'Former': 1,   # Former smoker (smoked 100+ cigarettes)
    'Current': 1   # Current smoker (smoked 100+ cigarettes)
}
df_lifestyle['Smoker'] = df_lifestyle['smoking_status'].map(smoking_map_lifestyle)

# > Cardiovascular/heart disease history...
df_lifestyle['Heart_Disease_or_Attack'] = df_lifestyle['cardiovascular_history']

# > Physical activity...
df_lifestyle['Physically_Active'] = (df_lifestyle['physical_activity_minutes_per_week'] > 0).astype(int)

# > Alcohol consumption...
df_lifestyle['Alcohol_Consumer'] = (df_lifestyle['alcohol_consumption_per_week'] > 0).astype(int)


# > Sex/Gender...
sex_map_indicators = {0: 'Female', 1: 'Male'}
df_indicators['Sex'] = df_indicators['Sex'].map(sex_map_indicators)

# > Blood Pressure - BINARY (0 = Normal, 1 = High)
# df_lifestyle: HighBP = 1 if DiastolicBP >= 80, else 0
df_lifestyle['HighBP'] = (df_lifestyle['DiastolicBP'] >= 80).astype(int)
# df_indicators: Already has HighBP column (0/1)
# df_healthcare: HighBP = 1 if DiastolicBP >= 80, else 0
df_healthcare['HighBP'] = (df_healthcare['DiastolicBP'] >= 80).astype(int)

# > Cholesterol - BINARY (0 = Normal, 1 = High)
# df_lifestyle: HighChol = 1 if Cholesterol >= 200, else 0
df_lifestyle['HighChol'] = (df_lifestyle['Cholesterol'] >= 200).astype(int)
# df_indicators: Already has HighChol column (0/1)
# ============================================================================




# ============================================================================
# ! Merging Datasets !
# ============================================================================
# Create df_common by merging df_lifestyle and df_indicators on BINARY columns only:
# This avoids Simpson's Paradox by using aligned binary columns
common_columns = ['BMI', 'Sex', 'Age_Group', 'Diabetes_Binary', 'Income_Level', 'Education_Level', 
                'Smoker', 'Heart_Disease_or_Attack', 'Physically_Active', 'Alcohol_Consumer', 
                'HighBP', 'HighChol']

df_lifestyle_common = df_lifestyle[common_columns].copy()
df_lifestyle_common['Source'] = 'lifestyle'

df_indicators_common = df_indicators[common_columns].copy()
df_indicators_common['Source'] = 'indicators'

df_common = pd.concat([df_lifestyle_common, df_indicators_common], ignore_index=True)

# Create df_common_general by merging all three datasets on matching BINARY columns only:
common_columns_general = ['BMI', 'Age_Group', 'Diabetes_Binary', 'HighBP']

df_lifestyle_common_general = df_lifestyle[common_columns_general].copy()
df_lifestyle_common_general['Source'] = 'lifestyle'

df_indicators_common_general = df_indicators[common_columns_general].copy()
df_indicators_common_general['Source'] = 'indicators'

df_healthcare_common_general = df_healthcare[common_columns_general].copy()
df_healthcare_common_general['Source'] = 'healthcare'

df_common_general = pd.concat([df_lifestyle_common_general, df_indicators_common_general, df_healthcare_common_general], ignore_index=True)
# ============================================================================




# ============================================================================
# ! VISUALIZATIONS - All 3 Datasets Combined !
# ============================================================================
# --- Figure 1: Age Group, Blood Pressure, and BMI (All 3 Datasets) ---
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Diabetes prevalence by Age Group
ax1a = axes1[0]
age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', 'Over 80']
diabetes_by_age = df_common_general.groupby('Age_Group')['Diabetes_Binary'].mean() * 100
diabetes_by_age = diabetes_by_age.reindex(age_order).dropna()
diabetes_by_age.plot(kind='bar', ax=ax1a, color='steelblue', edgecolor='black')
ax1a.set_title('Diabetes Prevalence by Age Group', fontsize=12, fontweight='bold')
ax1a.set_xlabel('Age Group')
ax1a.set_ylabel('Diabetes Prevalence (%)')
ax1a.tick_params(axis='x', rotation=45)

# 1b: Diabetes prevalence by HighBP
ax1b = axes1[1]
bp_labels = ['Normal BP', 'High BP']
diabetes_by_bp = df_common_general.groupby('HighBP')['Diabetes_Binary'].mean() * 100
colors_bp = ['lightblue', 'coral']
bars = ax1b.bar(bp_labels, diabetes_by_bp.values, color=colors_bp, edgecolor='black')
ax1b.set_title('Diabetes Prevalence by Blood Pressure', fontsize=12, fontweight='bold')
ax1b.set_xlabel('Blood Pressure Status')
ax1b.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_bp.values):
    ax1b.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

# 1c: BMI distribution by Diabetes status
ax1c = axes1[2]
df_no_diabetes = df_common_general[df_common_general['Diabetes_Binary'] == 0]['BMI']
df_diabetes = df_common_general[df_common_general['Diabetes_Binary'] == 1]['BMI']
bp_box = ax1c.boxplot([df_no_diabetes.dropna(), df_diabetes.dropna()], labels=['No Diabetes', 'Diabetes'], patch_artist=True)
bp_box['boxes'][0].set_facecolor('lightgreen')
bp_box['boxes'][1].set_facecolor('salmon')
ax1c.set_title('BMI Distribution by Diabetes Status', fontsize=12, fontweight='bold')
ax1c.set_xlabel('Diabetes Status')
ax1c.set_ylabel('BMI')

plt.suptitle('Key Health Indicators vs Diabetes Prevalence (All 3 Datasets)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./results/viz1_key_indicators.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================================




# ============================================================================
# ! VISUALIZATIONS - Lifestyle Factors (2 Datasets Only) !
# ============================================================================
# --- Figure 2: Lifestyle Factors (Smoker & Physically Active) ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 2a: Diabetes prevalence by Smoker status
ax2a = axes2[0]
smoker_labels = ['Non-Smoker', 'Smoker']
diabetes_by_smoker = df_common.groupby('Smoker')['Diabetes_Binary'].mean() * 100
colors_smoker = ['lightgreen', 'salmon']
bars = ax2a.bar(smoker_labels, diabetes_by_smoker.values, color=colors_smoker, edgecolor='black')
ax2a.set_title('Diabetes Prevalence by Smoking Status', fontsize=12, fontweight='bold')
ax2a.set_xlabel('Smoking Status')
ax2a.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_smoker.values):
    ax2a.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

# 2b: Diabetes prevalence by Physical Activity
ax2b = axes2[1]
phys_labels = ['Not Active', 'Physically Active']
diabetes_by_phys = df_common.groupby('Physically_Active')['Diabetes_Binary'].mean() * 100
colors_phys = ['salmon', 'lightgreen']
bars = ax2b.bar(phys_labels, diabetes_by_phys.values, color=colors_phys, edgecolor='black')
ax2b.set_title('Diabetes Prevalence by Physical Activity', fontsize=12, fontweight='bold')
ax2b.set_xlabel('Physical Activity Status')
ax2b.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_phys.values):
    ax2b.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

plt.suptitle('Lifestyle Factors: Smoking & Physical Activity vs Diabetes', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./results/viz2_lifestyle_factors.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 3: Additional Health Indicators (Cholesterol - 2 Datasets Only) - BINARY ---
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# 3a: Diabetes prevalence by HighChol (Binary: Normal vs High)
ax3a = axes3[0]
chol_labels = ['Normal Chol', 'High Chol']
diabetes_by_chol = df_common.groupby('HighChol')['Diabetes_Binary'].mean() * 100
colors_chol = ['lightgreen', 'coral']
bars = ax3a.bar(chol_labels, diabetes_by_chol.values, color=colors_chol, edgecolor='black')
ax3a.set_title('Diabetes Prevalence by Cholesterol Status', fontsize=12, fontweight='bold')
ax3a.set_xlabel('Cholesterol Status')
ax3a.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_chol.values):
    ax3a.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

# 3b: Diabetes prevalence by Heart Disease
ax3b = axes3[1]
heart_labels = ['No Heart Disease', 'Heart Disease']
diabetes_by_heart = df_common.groupby('Heart_Disease_or_Attack')['Diabetes_Binary'].mean() * 100
colors_heart = ['lightgreen', 'salmon']
bars = ax3b.bar(heart_labels, diabetes_by_heart.values, color=colors_heart, edgecolor='black')
ax3b.set_title('Diabetes Prevalence by Heart Disease History', fontsize=12, fontweight='bold')
ax3b.set_xlabel('Heart Disease Status')
ax3b.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_heart.values):
    ax3b.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

plt.suptitle('Health Indicators: Cholesterol & Heart Disease vs Diabetes', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./results/viz3_health_indicators.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================================




# ============================================================================
# ! VISUALIZATIONS - Surprising/Shocking Factors (2 Datasets Only) !
# ============================================================================
# --- Figure 4: Surprising Factors (Education Level & Income Level) ---
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

# 4a: Diabetes prevalence by Education Level
ax4a = axes4[0]
edu_order = ['No Formal', 'Elementary', 'Some High School', 'High School', 'Some College', 'College Graduate']
diabetes_by_edu = df_common.groupby('Education_Level')['Diabetes_Binary'].mean() * 100
diabetes_by_edu = diabetes_by_edu.reindex(edu_order).dropna()
bars = diabetes_by_edu.plot(kind='bar', ax=ax4a, color='mediumpurple', edgecolor='black')
ax4a.set_title('Diabetes Prevalence by Education Level', fontsize=12, fontweight='bold')
ax4a.set_xlabel('Education Level')
ax4a.set_ylabel('Diabetes Prevalence (%)')
ax4a.tick_params(axis='x', rotation=45)
for i, v in enumerate(diabetes_by_edu.values):
    ax4a.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

# 4b: Diabetes prevalence by Income Level
ax4b = axes4[1]
income_order = ['Low', 'Medium', 'High']
diabetes_by_income = df_common.groupby('Income_Level')['Diabetes_Binary'].mean() * 100
diabetes_by_income = diabetes_by_income.reindex(income_order).dropna()
bars = diabetes_by_income.plot(kind='bar', ax=ax4b, color='teal', edgecolor='black')
ax4b.set_title('Diabetes Prevalence by Income Level', fontsize=12, fontweight='bold')
ax4b.set_xlabel('Income Level')
ax4b.set_ylabel('Diabetes Prevalence (%)')
ax4b.tick_params(axis='x', rotation=0)
for i, v in enumerate(diabetes_by_income.values):
    ax4b.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('SURPRISING: Education & Income vs Diabetes Prevalence', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./results/viz4_surprising_factors.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================================