import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class RadiomicsModel:
    #----------Train Data ----------
    #Radiomics
    X_train_val = pd.read_csv("data/thymoma_radiomics_train.csv")
    #Clinical& Semantic, only done One-Hot Encoding for categorical variables.
    X_train_val_clinical = pd.read_csv("data/thymoma_clinical_semantic_ohe_train.csv")
    #Outcome
    y_train_val = X_train_val_clinical['WHO_Grade_Binary_Outcome']

    #----------Test Data----------
    #Radiomics
    X_test = pd.read_csv("data/thymoma_radiomics_test.csv")
    #Clinical& Semantic, only done One-Hot Encoding for categorical variables.
    X_test_clinical = pd.read_csv("data/thymoma_clinical_semantic_ohe_test.csv")
    #Outcome
    y_test = X_test_clinical['WHO_Grade_Binary_Outcome']

    # Outcome variables
    thymoma_outcomes = pd.concat([X_train_val_clinical, X_test_clinical]).sort_values(by='SubjectID')
    thymoma_outcomes = thymoma_outcomes[['SubjectID','WHO Grade Simplified', 'WHO_Grade_Binary_Outcome','Histopath report - WHO grade']]

    thymoma_outcomes.index = thymoma_outcomes.SubjectID.values
    thymoma_outcomes = thymoma_outcomes.drop(columns=["SubjectID"], axis=1)

    # Models
    r_features = (
        ['original_firstorder_90Percentile', 'original_shape_Sphericity'],
        [-1.180550982633071, -1.2101663996141763],
        -0.92761160498153
    )
    c_features = (
        ['AGE @ presentation', 'Anemia evaluation-PRCA_1.0'],
        [-4.137872075010222, -4.670739033901874],
        1.44798249199181

    )
    s_features = (
        ['TUMOR SHAPE_3.0', 'TUMOR CONTOUR_1.0', 'Areas of Nodular enhancement_1.0',
         'ATTENUATION & DEGREE OF ENHANCEMENT_3.0'],
        [1.590556710796678, -1.6596373212196924, -2.1405536851054734, -1.6414151374005408],
        0.425525399279825
    )

    train_proba = None
    train_distance_0 = None
    train_distance_1 = None
    train_distance_df= None
    X_train_val_normal = None

    thr = 0.36
    scalar =StandardScaler()

    def thymoma_model(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, np.array(self.r_features[1])) + self.r_features[2])))

    def trainmodel(self):
        self.X_train_val_normal  = self.scalar.fit_transform(self.X_train_val[self.r_features[0]])
        self.X_train_val_normal  = pd.DataFrame(self.X_train_val_normal, columns=self.r_features[0])

        dim_2 = []
        for i in range(self.X_train_val_normal.shape[0]):
            dim_1 = []
            for j in range(self.X_train_val_normal.shape[0]):
                d_of_x_y = (self.X_train_val_normal.iloc[j] - self.X_train_val_normal.iloc[i]).abs().sum()
                dim_1.append(d_of_x_y)
            dim_2.append(dim_1)

        self.train_distance_df = pd.DataFrame(dim_2, index=self.X_train_val.SubjectID.values, columns=self.X_train_val.SubjectID.values)
        self.train_distance_df["Outcome"] = self.y_train_val.values
        self.train_distance_df["Grades"] = self.X_train_val_clinical['WHO Grade Simplified'].values

        # Model
        self.train_proba = self.thymoma_model(self.X_train_val_normal)  #
        self.train_distance_0 = self.train_distance_df[self.train_distance_df.Outcome == 0]
        self.train_distance_1 = self.train_distance_df[self.train_distance_df.Outcome == 1]

    def determineUncertainties(self):
        train_uncertainties = []

        for id in self.train_distance_df.columns[:-2]:
            top_to_0 = self.train_distance_0.sort_values(id).iloc[1:2]
            top_to_1 = self.train_distance_1.sort_values(id).iloc[1:2]

            bottom_to_0 = self.train_distance_0.sort_values(id).iloc[-2:-1]
            bottom_to_1 = self.train_distance_1.sort_values(id).iloc[-2:-1]

            q1_0 = self.train_distance_0[id].quantile(0.25)
            q2_0 = self.train_distance_0[id].quantile(0.50)
            q3_0 = self.train_distance_0[id].quantile(0.75)
            iqr_0 = q3_0 - q1_0

            q1_1 = self.train_distance_1[id].quantile(0.25)
            q2_1 = self.train_distance_1[id].quantile(0.50)
            q3_1 = self.train_distance_1[id].quantile(0.75)
            iqr_1 = q3_1 - q1_1

            train_uncertainties.append((
                id,

                top_to_0.index[0],
                top_to_0[id].iloc[0],
                top_to_1.index[0],
                top_to_1[id].iloc[0],
                self.thymoma_outcomes.loc[top_to_0.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[top_to_1.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[top_to_0.index[0]]["Histopath report - WHO grade"],
                self.thymoma_outcomes.loc[top_to_1.index[0]]["Histopath report - WHO grade"],

                bottom_to_0.index[0],
                bottom_to_0[id].iloc[0],
                bottom_to_1.index[0],
                bottom_to_1[id].iloc[0],
                self.thymoma_outcomes.loc[bottom_to_0.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[bottom_to_1.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[bottom_to_0.index[0]]["Histopath report - WHO grade"],
                self.thymoma_outcomes.loc[bottom_to_1.index[0]]["Histopath report - WHO grade"],

                q1_0,
                q2_0,
                q3_0,
                iqr_0,

                q1_1,
                q2_1,
                q3_1,
                iqr_1,

            ))

        train_uncertainties = pd.DataFrame(
            train_uncertainties,
            columns=[
                "Train",
                "Near_0",
                "Near_0_Score",
                "Near_1",
                "Near_1_Score",
                "Near_0_Grade",
                "Near_1_Grade",
                "Near_0_WHO-Grade",
                "Near_1_WHO-Grade",

                "Far_0",
                "Far_0_Score",
                "Far_1",
                "Far_1_Score",
                "Far_0_Grade",
                "Far_1_Grade",
                "Far_0_WHO-Grade",
                "Far_1_WHO-Grade",

                "quartile_0 1",
                "quartile_0 2",
                "quartile_0 3",
                "iqr_0",

                "quartile_1 1",
                "quartile_1 2",
                "quartile_1 3",
                "iqr_1",

            ])

        train_uncertainties["Actual Grade"] = self.X_train_val_clinical['WHO Grade Simplified'].values
        uncercainty_scores_1 = []
        uncercainty_scores_0 = []
        for i, row in train_uncertainties.iterrows():
            uncercainty_scores_0.append(row.Near_0_Score * (row.Near_0_Score / row.Near_1_Score))
            uncercainty_scores_1.append(row.Near_1_Score * (row.Near_1_Score / row.Near_0_Score))

        train_uncertainties["Uncertainty_0"] = uncercainty_scores_0
        train_uncertainties["Uncertainty_1"] = uncercainty_scores_1
        train_uncertainties["Outcome"] = self.y_train_val.values
        train_uncertainties["Predicted"] = (self.train_proba >= self.thr).astype("int8").tolist()
        train_uncertainties["Proba_1"] = self.train_proba.tolist()
        train_uncertainties["LowUncertainty"] = [0 if v0 <= v1 else 1 for v0, v1 in
                                                 zip(uncercainty_scores_0, uncercainty_scores_1)]
        train_uncertainties["WHO-Grade"] = self.X_train_val_clinical["Histopath report - WHO grade"].values

        new_order = [
            "Train",
            "Near_0",
            "Near_0_Score",
            "Near_1",
            "Near_1_Score",
            "Outcome",
            "Predicted",
            "Proba_1",
            "Uncertainty_0",
            "Uncertainty_1",
            "LowUncertainty",
            "WHO-Grade",
            "Near_0_Grade",
            "Near_1_Grade",
            "Near_0_WHO-Grade",
            "Near_1_WHO-Grade",
            "Actual Grade",

            "Far_0",
            "Far_0_Score",
            "Far_1",
            "Far_1_Score",
            "Far_0_Grade",
            "Far_1_Grade",
            "Far_0_WHO-Grade",
            "Far_1_WHO-Grade",

            "quartile_0 1",
            "quartile_0 2",
            "quartile_0 3",
            "iqr_0",

            "quartile_1 1",
            "quartile_1 2",
            "quartile_1 3",
            "iqr_1", ]

    def test(self):
        X_test_normal = self.scalar.transform(self.X_test[self.r_features[0]])
        X_test_normal = pd.DataFrame(X_test_normal, columns=self.r_features[0])

        dim_2 = []
        for i in range(self.X_train_val_normal.shape[0]):
            dim_1 = []
            for j in range(X_test_normal.shape[0]):
                d_of_x_y = (X_test_normal.iloc[j] - self.X_train_val_normal.iloc[i]).abs().sum()
                dim_1.append(d_of_x_y)
            dim_2.append(dim_1)

        distance_df = pd.DataFrame(dim_2, index=self.X_train_val.SubjectID.values, columns=self.X_test.SubjectID.values)
        distance_df["Outcome"] = self.y_train_val.values
        distance_df["Grades"] = self.X_train_val_clinical['WHO Grade Simplified'].values

        # Model
        proba = self.thymoma_model(X_test_normal)  #
        distance_0 = distance_df[distance_df.Outcome == 0]
        distance_1 = distance_df[distance_df.Outcome == 1]

        uncertainties = []
        for id in distance_df.columns[:-2]:
            top_to_0 = distance_0.sort_values(id).iloc[:1]
            top_to_1 = distance_1.sort_values(id).iloc[:1]

            bottom_to_0 = distance_0.sort_values(id).iloc[-2:-1]
            bottom_to_1 = distance_1.sort_values(id).iloc[-2:-1]

            q1_0 = distance_0[id].quantile(0.25)
            q2_0 = distance_0[id].quantile(0.50)
            q3_0 = distance_0[id].quantile(0.75)
            iqr_0 = q3_0 - q1_0

            q1_1 = distance_1[id].quantile(0.25)
            q2_1 = distance_1[id].quantile(0.50)
            q3_1 = distance_1[id].quantile(0.75)
            iqr_1 = q3_1 - q1_1

            uncertainties.append((
                id,
                top_to_0.index[0],
                top_to_0[id].iloc[0],
                top_to_1.index[0],
                top_to_1[id].iloc[0],
                self.thymoma_outcomes.loc[top_to_0.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[top_to_1.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[top_to_0.index[0]]["Histopath report - WHO grade"],
                self.thymoma_outcomes.loc[top_to_1.index[0]]["Histopath report - WHO grade"],

                bottom_to_0.index[0],
                bottom_to_0[id].iloc[0],
                bottom_to_1.index[0],
                bottom_to_1[id].iloc[0],
                self.thymoma_outcomes.loc[bottom_to_0.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[bottom_to_1.index[0]]['WHO Grade Simplified'],
                self.thymoma_outcomes.loc[bottom_to_0.index[0]]["Histopath report - WHO grade"],
                self.thymoma_outcomes.loc[bottom_to_1.index[0]]["Histopath report - WHO grade"],

                q1_0,
                q2_0,
                q3_0,
                iqr_0,

                q1_1,
                q2_1,
                q3_1,
                iqr_1,
            ))

        uncertainties = pd.DataFrame(
            uncertainties,
            columns=[
                "Test",
                "Near_0",
                "Near_0_Score",
                "Near_1",
                "Near_1_Score",
                "Near_0_Grade",
                "Near_1_Grade",
                "Near_0_WHO-Grade",
                "Near_1_WHO-Grade",

                "Far_0",
                "Far_0_Score",
                "Far_1",
                "Far_1_Score",
                "Far_0_Grade",
                "Far_1_Grade",
                "Far_0_WHO-Grade",
                "Far_1_WHO-Grade",

                "quartile_0 1",
                "quartile_0 2",
                "quartile_0 3",
                "iqr_0",

                "quartile_1 1",
                "quartile_1 2",
                "quartile_1 3",
                "iqr_1",
            ])

        uncertainties["Actual Grade"] = self.X_test_clinical['WHO Grade Simplified'].values
        uncercainty_scores_1 = []
        uncercainty_scores_0 = []
        for i, row in uncertainties.iterrows():
            uncercainty_scores_0.append(row.Near_0_Score * (row.Near_0_Score / row.Near_1_Score))
            uncercainty_scores_1.append(row.Near_1_Score * (row.Near_1_Score / row.Near_0_Score))

        uncertainties["Uncertainty_0"] = uncercainty_scores_0
        uncertainties["Uncertainty_1"] = uncercainty_scores_1
        uncertainties["Outcome"] = self.y_test.values
        uncertainties["Predicted"] = (proba >= self.thr).astype("int8").tolist()
        uncertainties["Proba_1"] = proba.tolist()
        uncertainties["LowUncertainty"] = [0 if v0 <= v1 else 1 for v0, v1 in zip(uncercainty_scores_0, uncercainty_scores_1)]
        uncertainties["WHO-Grade"] = self.X_test_clinical["Histopath report - WHO grade"].values

        new_order = [
            "Test",
            "Near_0",
            "Near_0_Score",
            "Near_1",
            "Near_1_Score",
            "Outcome",
            "Predicted",
            "Proba_1",

            "Uncertainty_0",
            "Uncertainty_1",
            "LowUncertainty",
            "WHO-Grade",
            "Near_0_Grade",
            "Near_1_Grade",
            "Near_0_WHO-Grade",
            "Near_1_WHO-Grade",
            "Actual Grade",

            "Far_0",
            "Far_0_Score",
            "Far_1",
            "Far_1_Score",
            "Far_0_Grade",
            "Far_1_Grade",
            "Far_0_WHO-Grade",
            "Far_1_WHO-Grade",

            "quartile_0 1",
            "quartile_0 2",
            "quartile_0 3",
            "iqr_0",

            "quartile_1 1",
            "quartile_1 2",
            "quartile_1 3",
            "iqr_1", ]

        return uncertainties[new_order]

    def kNNDistance(self,k):
        X_test_normal = self.scalar.transform(self.X_test[self.r_features[0]])
        X_test_normal = pd.DataFrame(X_test_normal, columns=self.r_features[0])

        dim_2 = []
        for i in range(self.X_train_val_normal.shape[0]):
            dim_1 = []
            for j in range(X_test_normal.shape[0]):
                d_of_x_y = (X_test_normal.iloc[j] - self.X_train_val_normal.iloc[i]).abs().sum()
                dim_1.append(d_of_x_y)
            dim_2.append(dim_1)

        distance_df = pd.DataFrame(dim_2, index=self.X_train_val.SubjectID.values, columns=self.X_test.SubjectID.values)
        distance_df["Outcome"] = self.y_train_val.values
        distance_df["Grades"] = self.X_train_val_clinical['WHO Grade Simplified'].values

        proba = self.thymoma_model(X_test_normal)

        kNN= pd.DataFrame(columns=['ID', 'nn_0', 'nn_1', 'predicted', 'nn_agreement']);
        predicted = (proba >= self.thr).astype("int8").tolist()
        index = -1
        for i in distance_df:
            if(not ( i == 'Outcome' or i == 'Grades')):
                index+= 1
                sorted = distance_df[i].sort_values()
                selected = sorted[0:5]
                nn_0 = 0
                nn_1 = 0
                for k in selected.keys():
                    if distance_df['Outcome'][k] == 0:
                        nn_0+=1
                    else:
                        nn_1+=1
                pred = predicted[index]
                if pred == 1:
                    if nn_1 > nn_0:
                        nn_agreement = True
                    else:
                        nn_agreement = False
                else:
                    if nn_1 < nn_0:
                        nn_agreement = True
                    else:
                        nn_agreement = False
                newRow = pd.DataFrame([{'ID': i, 'nn_0': nn_0, 'nn_1':nn_1, 'predicted': pred, 'nn_agreement': nn_agreement}])
                kNN = pd.concat([kNN,newRow], ignore_index=True)

        return kNN

    def compareUncertainty(self, k):
        self.trainmodel()
        self.determineUncertainties()
        distance = self.test()

        kNN = self.kNNDistance(k)

        combined = pd.DataFrame(columns=['ID','True label', 'predicted', 'nn_0', 'nn_1',  'Uncertainty_0', 'Uncertainty_1', 'nn_agreement','uncertainty_agreement']);

        for i, row in kNN.iterrows():
            uncertainty_0 = distance['Uncertainty_0'][i]
            uncertainty_1 = distance['Uncertainty_1'][i]

            if kNN['predicted'][i] == 1 and uncertainty_1 < uncertainty_0:
                uncertainty_agreement = True
            elif kNN['predicted'][i] == 0 and uncertainty_0 < uncertainty_1:
                uncertainty_agreement = True
            else:
                uncertainty_agreement = False


            newRow = pd.DataFrame(
                [{'ID': kNN['ID'][i],'True label': distance['Outcome'][i], 'predicted': kNN['predicted'][i], 'nn_0': kNN['nn_0'][i], 'nn_1': kNN['nn_1'][i],
                  'Uncertainty_0': uncertainty_0, 'Uncertainty_1': uncertainty_1, 'nn_agreement': kNN['nn_agreement'][i], 'uncertainty_agreement':  uncertainty_agreement}])
            combined = pd.concat([combined,newRow], ignore_index=True)

        return combined
