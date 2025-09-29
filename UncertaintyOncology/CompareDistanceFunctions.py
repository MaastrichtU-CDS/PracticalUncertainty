from UncertaintyOncology.RadiomicsModel import RadiomicsModel

model = RadiomicsModel();
model.compareUncertainty(5).to_csv('results/comparison.csv', index=False)

