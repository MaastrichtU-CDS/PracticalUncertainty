import pandas

data_0 = pandas.read_csv("results/distance_0.csv")
data_0 = data_0.drop(data_0.columns[0], axis=1)
data_0 = data_0.drop('Outcome', axis=1)
data_0 = data_0.drop('Grades', axis=1)
data_1 = pandas.read_csv("results/distance_1.csv")
data_1 = data_1.drop(data_1.columns[0], axis=1)
data_1 = data_1.drop('Outcome', axis=1)
data_1 = data_1.drop('Grades', axis=1)

data_0 = data_0.replace(0.0, 1000)
data_1 = data_1.replace(0.0, 1000)

data_0['min_value'] = data_0.min(axis=1)
data_1['min_value'] = data_1.min(axis=1)

import matplotlib.pyplot as plt
low_risk = data_0['min_value']
high_risk = data_1['min_value']

box = plt.boxplot([low_risk, high_risk], labels = ["Low-Risk", "High-Risk"], patch_artist=True)

colors = ['red', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Labels and title
plt.ylabel("Distances to the nearest label")
plt.title("Box Plot of Risk Groups")

# Show plot
plt.show()
