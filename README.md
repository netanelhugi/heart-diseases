# Heart disease classification

In this project, we built 2 models to classify the presence of heart disease in patients:
1. Logistic regression.
2. Neural Network - The same model with more layers.

The model distinguishes between 2 options:
0 – Absence of heart disease.
1 – Presence of heart disease.
The data set we worked on contains files with patient details from 4 different hospitals,
consisting of 75 features.
We used only one data file (of Cleveland), and the 13 following features:
1. Age: in years.
2. Sex: 0=female, 1=male.
3. Chest pain type:
 - Value 1: typical angina.
 - Value 2: atypical angina.
 - Value 3: non-anginal pain.
 - Value 4: asymptomatic.
4. Resting blood pressure: in mm Hg.
5. Serum cholesterol: in mg/dl.
6. Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false.
7. Resting electrocardiographic results:
 - Value 0: normal.
 - Value 1: having ST-T wave abnormality.
 - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria.
8. Maximum heart rate achieved.
9. Exercise induced angina: 1 = yes, 0 = no.
10. ST depression induced by exercise relative to rest.
11. The slope of the peak exercise ST segment:
 - Value 1: upsloping.
 - Value 2: flat.
 - Value 3: downsloping.
12. Number of major vessels: 0/1/2/3.
13. Thal: 3 = normal, 6 = fixed defect, 7 = reversable defect.

(Link to full description: https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
The file of Cleveland database contains data of 303 different patients.
From this data we used 70% for train data, and the rest to testing our model.
The data was divided between test and train randomly.

**For model results, refer to "Assignment description.pdf"
In both folders.**
