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

# > Clean datasets by removing NaN values from key columns
df_lifestyle = df_lifestyle.dropna()
df_indicators = df_indicators.dropna()
df_healthcare = df_healthcare.dropna()
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
# ! VISUALIZATIONS !
# ============================================================================

# ============================================================================
# ! SECTION 1: Overview - The Big Picture !
# ============================================================================
# --- Figure 1: Pie Chart - Overall Diabetes Prevalence ---
fig1, ax1 = plt.subplots(figsize=(8, 8))
diabetes_counts = df_common_general['Diabetes_Binary'].value_counts()
labels_pie = ['No Diabetes', 'Diabetes']
colors_pie = ['mediumaquamarine', 'coral']
explode_pie = (0, 0.05)  # Slightly explode the Diabetes slice
ax1.pie(diabetes_counts, labels=labels_pie, colors=colors_pie, explode=explode_pie,
        autopct='%1.1f%%', startangle=90, shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Overall Diabetes Prevalence', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/viz1_pie_diabetes_prevalence.png', dpi=150, bbox_inches='tight')
# ============================================================================




# ============================================================================
# ! SECTION 2: Demographics - Age & BMI !
# ============================================================================
# --- Figure 2: Diabetes prevalence by Age Group ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', 'Over 80']
diabetes_by_age = df_common_general.groupby('Age_Group')['Diabetes_Binary'].mean() * 100
diabetes_by_age = diabetes_by_age.reindex(age_order)
diabetes_by_age.plot(kind='bar', ax=ax2, color='royalblue', edgecolor='black')
ax2.set_title('Diabetes Prevalence by Age Group', fontsize=14, fontweight='bold')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Diabetes Prevalence (%)')
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('./results/viz2_age_group.png', dpi=150, bbox_inches='tight')

# --- Figure 3: BMI distribution by Diabetes status ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
df_no_diabetes = df_common_general[df_common_general['Diabetes_Binary'] == 0]['BMI']
df_diabetes = df_common_general[df_common_general['Diabetes_Binary'] == 1]['BMI']
bp_box = ax3.boxplot([df_no_diabetes, df_diabetes], labels=['No Diabetes', 'Diabetes'], patch_artist=True)
bp_box['boxes'][0].set_facecolor('palegreen')
bp_box['boxes'][1].set_facecolor('red')
ax3.set_title('BMI Distribution by Diabetes Status', fontsize=14, fontweight='bold')
ax3.set_xlabel('Diabetes Status')
ax3.set_ylabel('BMI')
plt.tight_layout()
plt.savefig('./results/viz3_bmi_distribution.png', dpi=150, bbox_inches='tight')
# ============================================================================




# ============================================================================
# ! SECTION 3: Health Indicators - BP, Cholesterol, Heart Disease !
# ============================================================================
# --- Figure 4: Diabetes prevalence by Blood Pressure ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
bp_labels = ['Normal/Low BP', 'High BP']
diabetes_by_bp = df_common_general.groupby('HighBP')['Diabetes_Binary'].mean() * 100
colors_bp = ['lightblue', 'orange']
bars = ax4.bar(bp_labels, diabetes_by_bp.values, color=colors_bp, edgecolor='black')
ax4.set_title('Diabetes Prevalence by Blood Pressure', fontsize=14, fontweight='bold')
ax4.set_xlabel('Blood Pressure Status')
ax4.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_bp.values):
    ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('./results/viz4_blood_pressure.png', dpi=150, bbox_inches='tight')

# --- Figure 5: Diabetes prevalence by Cholesterol Status ---
fig5, ax5 = plt.subplots(figsize=(8, 6))
chol_labels = ['Normal/Low Cholesterol', 'High Cholesterol']
diabetes_by_chol = df_common.groupby('HighChol')['Diabetes_Binary'].mean() * 100
colors_chol = ['darkseagreen', 'salmon']
bars = ax5.bar(chol_labels, diabetes_by_chol.values, color=colors_chol, edgecolor='black')
ax5.set_title('Diabetes Prevalence by Cholesterol Status', fontsize=14, fontweight='bold')
ax5.set_xlabel('Cholesterol Status')
ax5.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_chol.values):
    ax5.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('./results/viz5_cholesterol_status.png', dpi=150, bbox_inches='tight')

# --- Figure 6: Diabetes prevalence by Heart Disease ---
fig6, ax6 = plt.subplots(figsize=(8, 6))
heart_labels = ['No Heart Disease/Heart Attack', 'Heart Disease/Heart Attack']
diabetes_by_heart = df_common.groupby('Heart_Disease_or_Attack')['Diabetes_Binary'].mean() * 100
colors_heart = ['cadetblue', 'lightcoral']
bars = ax6.bar(heart_labels, diabetes_by_heart.values, color=colors_heart, edgecolor='black')
ax6.set_title('Diabetes Prevalence by Heart Disease/Heart Attack History', fontsize=14, fontweight='bold')
ax6.set_xlabel('Heart Disease/Heart Attack Status')
ax6.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_heart.values):
    ax6.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('./results/viz6_heart_disease.png', dpi=150, bbox_inches='tight')
# ============================================================================




# ============================================================================
# ! SECTION 4: Lifestyle Factors - Smoking & Physical Activity !
# ============================================================================
# --- Figure 7: Diabetes prevalence by Smoking Status ---
fig7, ax7 = plt.subplots(figsize=(8, 6))
smoker_labels = ['Non-Smoker', 'Smoker']
diabetes_by_smoker = df_common.groupby('Smoker')['Diabetes_Binary'].mean() * 100
colors_smoker = ['mediumseagreen', 'firebrick']
bars = ax7.bar(smoker_labels, diabetes_by_smoker.values, color=colors_smoker, edgecolor='black')
ax7.set_title('Diabetes Prevalence by Smoking Status', fontsize=14, fontweight='bold')
ax7.set_xlabel('Smoking Status')
ax7.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_smoker.values):
    ax7.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('./results/viz7_smoking_status.png', dpi=150, bbox_inches='tight')

# --- Figure 8: Diabetes prevalence by Physical Activity ---
fig8, ax8 = plt.subplots(figsize=(8, 6))
phys_labels = ['Not Active', 'Physically Active']
diabetes_by_phys = df_common.groupby('Physically_Active')['Diabetes_Binary'].mean() * 100
colors_phys = ['indianred', 'limegreen']
bars = ax8.bar(phys_labels, diabetes_by_phys.values, color=colors_phys, edgecolor='black')
ax8.set_title('Diabetes Prevalence by Physical Activity', fontsize=14, fontweight='bold')
ax8.set_xlabel('Physical Activity Status')
ax8.set_ylabel('Diabetes Prevalence (%)')
for i, v in enumerate(diabetes_by_phys.values):
    ax8.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('./results/viz8_physical_activity.png', dpi=150, bbox_inches='tight')
# ============================================================================




# ============================================================================
# ! SECTION 5: Socioeconomic Factors - Education & Income !
# ============================================================================
# --- Figure 9: Diabetes prevalence by Education Level ---
fig9, ax9 = plt.subplots(figsize=(10, 6))
edu_order = ['No Formal', 'Elementary', 'Some High School', 'High School', 'Some College', 'College Graduate']
diabetes_by_edu = df_common.groupby('Education_Level')['Diabetes_Binary'].mean() * 100
diabetes_by_edu = diabetes_by_edu.reindex(edu_order)
bars = diabetes_by_edu.plot(kind='bar', ax=ax9, color='slateblue', edgecolor='black')
ax9.set_title('Diabetes Prevalence by Education Level', fontsize=14, fontweight='bold')
ax9.set_xlabel('Education Level')
ax9.set_ylabel('Diabetes Prevalence (%)')
ax9.tick_params(axis='x', rotation=45)
for i, v in enumerate(diabetes_by_edu.values):
    ax9.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('./results/viz9_education_level.png', dpi=150, bbox_inches='tight')

# --- Figure 10: Diabetes prevalence by Income Level ---
fig10, ax10 = plt.subplots(figsize=(8, 6))
income_order = ['Low', 'Medium', 'High']
diabetes_by_income = df_common.groupby('Income_Level')['Diabetes_Binary'].mean() * 100
diabetes_by_income = diabetes_by_income.reindex(income_order)
bars = diabetes_by_income.plot(kind='bar', ax=ax10, color='darkcyan', edgecolor='black')
ax10.set_xticklabels(['Lower Class', 'Middle Class', 'Upper Class'])
ax10.set_title('Diabetes Prevalence by Income Level', fontsize=14, fontweight='bold')
ax10.set_xlabel('Income Level')
ax10.set_ylabel('Diabetes Prevalence (%)')
ax10.tick_params(axis='x', rotation=0)
for i, v in enumerate(diabetes_by_income.values):
    ax10.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('./results/viz10_income_level.png', dpi=150, bbox_inches='tight')
# ============================================================================




# ============================================================================
# ! SECTION 6: Combined Risk Factor Analysis !
# ============================================================================
# --- Figure 11: Combined Risk Factors (BP & Cholesterol) ---
fig11, ax11 = plt.subplots(figsize=(12, 7))

# Create risk factor combinations
df_risk = df_common.copy()
df_risk['Risk_Category'] = 'No Risk Factors'
df_risk.loc[(df_risk['HighBP'] == 1) & (df_risk['HighChol'] == 0), 'Risk_Category'] = 'High BP Only'
df_risk.loc[(df_risk['HighBP'] == 0) & (df_risk['HighChol'] == 1), 'Risk_Category'] = 'High Cholesterol Only'
df_risk.loc[(df_risk['HighBP'] == 1) & (df_risk['HighChol'] == 1), 'Risk_Category'] = 'Both High BP & Cholesterol'

# Calculate diabetes prevalence for each risk category
risk_order = ['No Risk Factors', 'High BP Only', 'High Cholesterol Only', 'Both High BP & Cholesterol']
diabetes_by_risk = df_risk.groupby('Risk_Category')['Diabetes_Binary'].mean() * 100
diabetes_by_risk = diabetes_by_risk.reindex(risk_order)

# Create gradient colors from green (low risk) to red (high risk)
colors_risk = ['seagreen', 'goldenrod', 'darkorange', 'darkred']
bars = ax11.bar(risk_order, diabetes_by_risk.values, color=colors_risk, edgecolor='black', width=0.6)

ax11.set_title('Diabetes Prevalence by Combined Risk Factors\n(High Blood Pressure & High Cholesterol)', fontsize=14, fontweight='bold')
ax11.set_xlabel('Risk Factor Combination', fontsize=12)
ax11.set_ylabel('Diabetes Prevalence (%)', fontsize=12)
ax11.tick_params(axis='x', rotation=15)

# Add value labels on bars
for i, v in enumerate(diabetes_by_risk.values):
    ax11.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add a horizontal line showing overall average
avg_prevalence = df_risk['Diabetes_Binary'].mean() * 100
ax11.axhline(y=avg_prevalence, color='gray', linestyle='--', linewidth=1.5, label=f'Overall Average: {avg_prevalence:.1f}%')
ax11.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./results/viz11_combined_bp_cholesterol.png', dpi=150, bbox_inches='tight')

# --- Figure 12: Stacked Bar Chart - Cumulative Risk Factor Count ---
fig12, ax12 = plt.subplots(figsize=(10, 7))

# Count how many risk factors each person has (HighBP, HighChol, Smoker, Not Physically Active)
df_risk_count = df_common.copy()
df_risk_count['Not_Active'] = (df_risk_count['Physically_Active'] == 0).astype(int)
df_risk_count['Risk_Factor_Count'] = (df_risk_count['HighBP'] + df_risk_count['HighChol'] + 
                                    df_risk_count['Smoker'] + df_risk_count['Not_Active'])

# Group by risk factor count and diabetes status
risk_count_diabetes = df_risk_count.groupby(['Risk_Factor_Count', 'Diabetes_Binary']).size().unstack(fill_value=0)

# Calculate percentages for stacking
risk_count_pct = risk_count_diabetes.div(risk_count_diabetes.sum(axis=1), axis=0) * 100

# Create stacked bar chart
x_labels = ['0 Risk Factors', '1 Risk Factor', '2 Risk Factors', '3 Risk Factors', '4 Risk Factors']
x_positions = range(len(risk_count_pct))
bottom_vals = risk_count_pct[0].values  # No Diabetes
top_vals = risk_count_pct[1].values     # Diabetes

bars1 = ax12.bar(x_positions, bottom_vals, color='mediumseagreen', edgecolor='black', label='No Diabetes')
bars2 = ax12.bar(x_positions, top_vals, bottom=bottom_vals, color='tomato', edgecolor='black', label='Diabetes')

# Add percentage labels on the diabetes portion
for i, (b, t) in enumerate(zip(bottom_vals, top_vals)):
    ax12.text(i, b + t/2, f'{t:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax12.set_xticks(x_positions)
ax12.set_xticklabels(x_labels)
ax12.set_xlabel('Number of Risk Factors\n(High BP, High Cholesterol, Smoker, Physically Inactive)', fontsize=11)
ax12.set_ylabel('Percentage (%)', fontsize=12)
ax12.set_title('Diabetes Prevalence by Cumulative Risk Factor Count', fontsize=14, fontweight='bold')
ax12.legend(loc='upper left')
ax12.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('./results/viz12_cumulative_risk_factors.png', dpi=150, bbox_inches='tight')
# ============================================================================