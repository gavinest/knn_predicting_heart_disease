<center><h1> Predicting Heart Disease </h1> <h4>using KNearestNeighbors classifier</h4></center>

---
In this sprint, I use a KNearestNeighbors classifier to predict the presence of heart disease in patients from the [UCI processed Cleveland dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).

The baseline model with k=1 Nearest neighbor predicted at 58% accuracy. Below I walk you through my process of improving the model to 89% accuracy at the end of the sprint.

---
# Exploratory Data Analysis

The Jupyter (Ipython) notebook [(eda.ipynb)](./eda.ipynb) contains my code for this section.

_Findings:_  
Addressed the challenge of finding what each column corresponded to and which was the predictor column through some [Googling](http://archive.ics.uci.edu/ml/datasets/Heart+Disease).

EDA revealed missing data in 2 columns. I explored saving these data rows by replacing missing data with the feature’s mode (since these were categorical features). However this made the model predict worse. I ended up dropping the rows with missing data, loosing 6 of the 303 datapoints.

I also plotted distributions. I cleaned the data so the data types aligned for the analysis portion.

The predictor column (‘num’) contained 5 categories (0-4) making it difficult to perform a classification. Also the distribution of the predictor column showed extreme class imbalance. I found [instructions]((http://archive.ics.uci.edu/ml/datasets/Heart+Disease) that ‘0’ value indicates to presence of heart disease while 1-4 indicates possibility of heart disease. Therefore I mapped boolean values to this column which changed the 1-4 values to ‘1’.  Now this became a classification problem of identifying patient as not having heart disease (0 prediction) or having the presence of heart disease (1 prediction). Doing this also addressed the previous class imbalance problem so I no longer needed to explore options to address this such as stratification, oversampling, or synthetic minority oversampling technique (SMOTE).

Distribution of predictor ('num') before and after mapping boolean columns. ![boolean_map](./imgs/predictor_columns.png)
