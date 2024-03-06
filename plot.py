import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data which is an excel file
df = pd.read_excel('Melanoma Results.xlsx')

# Update the row model names under the column "Model" to: # Plot the data ["MeshNet", "ResNet152(+ISIC19)", "DenseNet121", "ResNet50(+ISIC18)", "ResNet152(+combined)", "ResNet50(+ISIC20)"]
df["Model"] = ["MeshNet", "ResNet152\n(+ISIC19)", "DenseNet121", "ResNet50\n(+ISIC18)", "ResNet152\n(+combined)", "ResNet50\n(+ISIC20)"]

# Sort the data based on the "Avg" column
df = df.sort_values(by="Avg")

# Divide the avg time by 1000 to get the seconds
df["Avg"] = df["Avg"]/1000

plt.style.use('grayscale')

# Bar charrt
fig, ax = plt.subplots()

barWidth = 0.25

# Make the plot
plt.bar(df["Model"], df["Avg"], width=barWidth, label='Mean Scripting Time')

# Add the values of the seconds on top of each bar
for i, v in enumerate(df["Avg"]):
    plt.text(i, v + 0.1, str(round(v, 2)), ha='center')

# Add xticks on the middle of the group bars
plt.xlabel('Best Performing Models & Meshnet', fontweight='bold')
plt.ylabel('Scripting Time (s)', fontweight='bold')

# Create legend & Show graphic
plt.legend()
plt.style.use('grayscale')

# Add padding at the bottom
plt.tight_layout()
plt.savefig('test.png')
plt.show()