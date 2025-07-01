import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
exercise = pd.read_csv("exercise.csv")

# Set the figure size and style
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Create the violin plot with overlayed swarm plot
sns.violinplot(x="kind", y="pulse", data=exercise, palette="Set3", inner=None)
sns.swarmplot(x="kind", y="pulse", data=exercise, color="black", size=3)

# Add title and labels
plt.title("Pulse Distribution by Type of Exercise", fontsize=16)
plt.xlabel("Type of Exercise", fontsize=12)
plt.ylabel("Pulse Rate", fontsize=12)

# Improve layout
plt.tight_layout()
plt.show()