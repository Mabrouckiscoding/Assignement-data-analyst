# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"Assignement-data-analyst/healthcare_dataset.csv"
df = pd.read_csv(file_path)

# Group data by Blood Group, Gender, and Medical Condition and count occurrences
correlation_data = df.groupby(['Blood Group Type', 'Gender', 'Medical Condition']).size().reset_index(name='Count')

# Pivot the data for heatmap
pivot_table = correlation_data.pivot_table(
    index=['Blood Group Type', 'Gender'],
    columns='Medical Condition',
    values='Count',
    fill_value=0
)

# Create the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
plt.title('Medical Conditions by Blood Group and Gender')
plt.ylabel('Blood Group & Gender')
plt.xlabel('Medical Condition')
plt.tight_layout()
plt.show()

# Calculate Average Billing Amount for each Medical Condition
avg_billing = df.groupby('Medical Condition')['Billing Amount'].mean().sort_values(ascending=False).round()

# Convert to a DataFrame for easy viewing or export
avg_billing_df = avg_billing.reset_index()
avg_billing_df.columns = ['Medical Condition', 'Average Billing Amount']

# Print the results
print("\nAverage Billing Amount per Medical Condition:\n")
print(avg_billing_df)
