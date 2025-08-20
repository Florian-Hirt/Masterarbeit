import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set default font size
plt.rcParams.update({'font.size': 12})

# Data from the table
data = {
    'Model': ['Gemma 3 4B', 'Gemma 3 12B', 'Llama 3.1 8B', 'Llama 3.2 3B'],
    'Model_Short': ['Gemma\n3 4B', 'Gemma\n3 12B', 'Llama\n3.1 8B', 'Llama\n3.2 3B'],  # Line breaks for cleaner display
    'Model_Abbrev': ['G3-4B', 'G3-12B', 'L3.1-8B', 'L3.2-3B'],  # Very short abbreviations
    'Orig_Avg': [2.70, 3.25, 2.95, 2.95],
    'Adv_Avg': [2.46, 2.64, 2.38, 2.32],
    'Avg_Drop': [0.24, 0.61, 0.57, 0.62],
    'Pct_Worse': [27.0, 52.7, 43.2, 51.4]
}

df = pd.DataFrame(data)

# Create figure with larger size
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Model Performance Under Adversarial Conditions', fontsize=20, fontweight='bold')

# 1. Grouped bar chart: Original vs Adversarial Average
x = np.arange(len(df['Model']))
width = 0.35

bars1 = ax1.bar(x - width/2, df['Orig_Avg'], width, label='Original Avg', color='skyblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['Adv_Avg'], width, label='Adversarial Avg', color='lightcoral', alpha=0.8)

ax1.set_xlabel('Models', fontsize=14)
ax1.set_ylabel('Average Score', fontsize=14)
ax1.set_title('Original vs Adversarial Performance', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model_Short'], fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelsize=12)

# Add value labels on bars with larger font
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# 2. Average Drop chart
bars3 = ax2.bar(df['Model_Short'], df['Avg_Drop'], color='orange', alpha=0.7)
ax2.set_xlabel('Models', fontsize=14)
ax2.set_ylabel('Average Drop', fontsize=14)
ax2.set_title('Performance Drop Under Adversarial Conditions', fontsize=16)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(True, alpha=0.3)

# Add value labels with larger font
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# 3. Percentage of samples that got worse
bars4 = ax3.bar(df['Model_Short'], df['Pct_Worse'], color='red', alpha=0.6)
ax3.set_xlabel('Models', fontsize=14)
ax3.set_ylabel('% Samples Degraded', fontsize=14)
ax3.set_title('Percentage of Samples That Got Worse', fontsize=16)
ax3.tick_params(axis='both', labelsize=12)
ax3.grid(True, alpha=0.3)

# Add value labels with larger font
for bar in bars4:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# 4. Scatter plot: Avg Drop vs % Samples Worse
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ax4.scatter(df['Avg_Drop'], df['Pct_Worse'], s=150, alpha=0.7, c=colors)
for i, model in enumerate(df['Model_Abbrev']):
    # Use shorter names for scatter plot annotations
    short_name = model.replace(' ', '\n', 1)  # Split at first space
    ax4.annotate(short_name, (df['Avg_Drop'][i], df['Pct_Worse'][i]),
                xytext=(8, 8), textcoords='offset points', fontsize=11, ha='left')

ax4.set_xlabel('Average Drop', fontsize=14)
ax4.set_ylabel('% Samples That Got Worse', fontsize=14)
ax4.set_title('Relationship: Performance Drop vs Sample Degradation', fontsize=16)
ax4.tick_params(axis='both', labelsize=12)
ax4.grid(True, alpha=0.3)

# Set x-axis limits to give more space for annotations
ax4.set_xlim(0.2, 0.7)

# Adjust layout to prevent overlap
plt.tight_layout()

# Use LaTeX fonts to match document
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

# Save the figure with high DPI for better quality
output_path = 'model_performance.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()