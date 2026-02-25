import pandas

data = pandas.read_csv("results/comparison.csv")

intra_distance_0 = data.loc[data['True label'] == 0, 'Uncertainty_0'].mean()
intra_distance_1 = data.loc[data['True label'] == 1, 'Uncertainty_1'].mean()
inter_distance_0 = data.loc[data['True label'] == 0, 'Uncertainty_1'].mean()
inter_distance_1 = data.loc[data['True label'] == 1, 'Uncertainty_0'].mean()

print(intra_distance_0)
print(intra_distance_1)
print(inter_distance_0)
print(inter_distance_1)