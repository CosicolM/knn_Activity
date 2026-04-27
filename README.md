# knn_Activity
HOMEWORK: K-Nearest Neighbors (KNN) on Diabetes Dataset 
Course: Computational Science for Computer Science 
Topic: Machine Learning – K-Nearest Neighbors 

Objective
This homework aims to develop your understanding of instance-based learning by applying the K-Nearest Neighbors (KNN) algorithm to a real-world dataset. You will analyze, preprocess, and classify data to predict diabetes outcomes.
 Dataset Description 
You are provided with a dataset containing 768 patient records with the following features:
• Pregnancies
• Glucose 
• BloodPressure 
• SkinThickness 
• Insulin
• BMI 
• DiabetesPedigreeFunction 
• Age 
• Outcome (Target Variable: 0 = Non-diabetic, 1 = Diabetic)

TASK REQUIREMENTS
Part 1: Data Understanding
	Describe each feature in your own words.
	Pregnancies 
	Number of times the patient has been pregnant. This is relevant because pregnancy can influence hormonal balance and diabetes risk.
	Glucose
	Blood glucose level (usually fasting). This is a direct indicator of how well the body regulates sugar – high values strongly suggest diabetes.
	BloodPressure
	Diastolic blood pressure (lower number in a BP reading). High blood pressure is often associated with metabolic disorders, including diabetes.
	SkinThickness
	Thickness of the skin fold (usually measured at the triceps). It’s used as an indirect measure of body fat.
	Insulin
	Insulin level in the blood. Insulin regulates blood sugar, abnormal levels may indicate insulin resistance or diabetes.
	BMI
	A measure of body fat based on height and weight. Higher BMI often correlates with increased diabetes risk.
	DiabetesPedigreeFunction
	A score representing genetic likelihood of diabetes based on family history. Higher values suggest stronger hereditary risk.
	Age
	Age of the patient. Risk of diabetes generally increases with age.
	Outcome
	0 = Non-diabetic
	1 = Diabetic
	Identify:
	Which features are likely most important for prediction
                     This features are known major risk factors for diabetes:
	Glucose – most important (direct measure of diabetes condition)
	BMI – Strong link with obesity and diabetes
	Age – Risk increases over time
	Insulin  – Indicates how the body processes glucose
	DiabetesPedigreeFunction – Capture genetic risk

	Which features may contain missing or problematic values
	Glucose
	BloodPressure
	SkinThickness
	Insulin
	BMI
Part 2: Data Preprocessing(20 pts)
	Check for missing or zero values
	Glucose
	BloodPressure
	SkinThickness
	Insulin
	BMI
  <img width="411" height="210" alt="image" src="https://github.com/user-attachments/assets/49d51dd2-514b-4c4b-a09e-b2795c1ee50a" />


	Handle missing data using one of the following: 
	Mean/median imputation
	Median Imputation – is a way of filling missing (or invalid) values using the median of each column.
To get the median of a dataset, add all valid  numbers in a column then sort in ascending order. After that, find the middle values, add then divide it to 2.

	Glucose = 117
	BloodPressure = 72
	SkinThickness = 23
	Insulin = 30.5
	BMI = 32

    
	3. Apply feature scaling (required for KNN):
	Standardization or Min-Max normalization
	I uses Standardization because it ensure that all features contribute equally to the distance calculation in KNN. Since KNN is a distance-based algorithm, features with larger numeric ranges (such as Glucose or BMI) it would otherwise dominate the model. By transforming tha data to have a mean of 0 and standard deviation of 1, standardization ensures fair comparison between features and improves model performance.
	4. Show before and after preprocessing results (tables or summary statistics).
<img width="1039" height="199" alt="image" src="https://github.com/user-attachments/assets/f4a53c85-710c-4181-a634-47b0f6b8adb7" />
  <img width="1067" height="185" alt="image" src="https://github.com/user-attachments/assets/c4a2f87f-e8f7-467b-8565-4081b9ba0aec" />
After scaling, all features were transformed using StandardScaler, which converts values into a standardized form where the mean is 0 and standard deviation is 1. Positive values indicate above-average measurements, while negative values indicate below-average measurements relative to the dataset.
PART 3:
1. Split the dataset: 10
80% Training = 8
20% Testing = 2
<img width="255" height="137" alt="image" src="https://github.com/user-attachments/assets/def2c34a-ddf6-479a-b3fa-77170d9d4666" />

 Training Row 1
(1, 85, 66, 29, 0, 26.6, 0.351, 31)
d = √ [(6−1)² + (148−85)² + (72−66)² + (35−29)² + (0−0)² + (33.6−26.6)² + (0.627−0.351)² + (50−31)²]
= √[25 + 3969 + 36 + 36 + 0 + 49 + 0.076 + 361]
= √4376.076
 = 66.14

Training Row 2
(8, 183, 64, 0, 0, 23.3, 0.672, 32)
= √[(6−8)² + (148−183)² + (72−64)² + (35−0)² + (0−0)² + (33.6−23.3)² + (0.627−0.672)² + (50−32)²]
= √[4 + 1225 + 64 + 1225 + 0 + 106.09 + 0.002 + 324]
= √2948.092
 = 54.30

Training Row 3
(1, 89, 66, 23, 94, 28.1, 0.167, 21)
= √[25 + 3481 + 36 + 144 + 8836 + 30.25 + 0.211 + 841]
= √13393.461
 = 115.74

Training Row 4
(0, 137, 40, 35, 168, 43.1, 2.288, 33)
= √[36 + 121 + 1024 + 0 + 28224 + 90.25 + 2.77 + 289 ]
= √29786.02
 = 172.59

 Training Row 5
(5, 116, 74, 0, 0, 25.6, 0.201, 30)
= √[1 + 1024 + 4 + 1225 + 0 + 64 + 0.181 + 400]
= √2718.181
 = 52.13

 Training Row 6
(3, 78, 50, 32, 88, 31, 0.248, 26)
= √[9 + 4900 + 484 + 9 + 7744 + 6.76 + 0.144 + 576]
= √13728.904
 = 117.19

Training Row 7
(10, 115, 0, 0, 0, 35.3, 0.134, 29)
= √[16 + 1089 + 5184 + 1225 + 0 + 2.89 + 0.243 + 441 ]
= √7958.133
 = 89.21

 Training Row 8
(2, 197, 70, 45, 543, 30.5, 0.158, 53)
= √ [0 + 0 + 0 + 100 + 294849 + 9.0 + 0 + 0]
= √294958
 = 543.10

 Training Row 9
(8, 125, 96, 0, 0, 0, 0.232, 54)
= √[4 + 529 + 576 + 1225 + 0 + 1128.96 + 0.005 + 16]
= √3478.965
 = 58.99

Training Row 10
(4, 110, 92, 0, 0, 37.6, 0.191, 30)
= √[(6−4)² + (148−110)² + (72−92)² + (35−0)² + (0−0)² + (33.6−37.6)² + (0.627−0.191)² + (50−30)²]
= √[4 + 1444 + 400 + 1225 + 0 + 16 + 0.190 + 400]
= √3489.190
 = 59.07
<img width="493" height="701" alt="image" src="https://github.com/user-attachments/assets/f0ba90e3-a9f4-4eed-9c3c-3025ebb639eb" />

<img width="647" height="601" alt="image" src="https://github.com/user-attachments/assets/bf8e3f43-2a3e-4497-b73b-4297c007bde4" />

Part 4: Model Evaluation
Evaluate your model using:
	-Accuracy
	-Confusion Matrix 
<img width="226" height="112" alt="image" src="https://github.com/user-attachments/assets/c699f525-6570-401d-96f0-d06b805e343a" />

This means:
	83 = True Negatives (correctly predicted non-diabetic)
	25 = False Positives ( predicted diabetic but actually not)
	18 = False Negatives ( missed diabetic cases)
	28 = True Positives ( correctly predicted diabetic)

The values in the confusion matrix are derived from predictions on the full test dataset. Since 20% of the 768 records were used for testing, the test set contains approximately 154 samples. The confusion matrix entries (83, 25, 18, 28) represent counts of correct and incorrect classifications across all these test instances. Specifically, 83 and 28 correspond to correct predictions (true negatives and true positives), while 25 and 18 represent misclassifications (false positives and false negatives).

Accuracy is calculated by dividing the number of correct predictions by the total number of predictions. In this case, the number of correct predictions is 83 + 28 = 111, and the total number of predictions is 154. Therefore, the accuracy is:

Accuracy = (TP+TN)/Total
                = 28 + 83 / 83+ 25+ 18+ 28 = 111/154 = 0.7208

This means the model correctly classified about 72% of the test instances.


Answer the following:
	1. Which value of K performed best?
K = 3 performed best because it correctly classified the test instance while K = 5 and K = 7 misclassified it.

	2. Why does performance change with different K values?
Because K controls how many neighbors vote.
	Small K focuses more distant neighbors
	Large K includes more distant neighbors 
Those extra neighbors can belong to another class and override the correct prediction. Distance stops mattering and majority takes over, which sometimes ruins everything.

	3. What happens when K is too small or too large? 
	K too small (like 1 or 3):
Very sensitive to noise and outliers. One weird data point can hijack the prediction. 
	K too large.
The model becomes too general. It starts averaging everything and ignores local patterns. Eventually it predicts the majority class like a lazy student guessing “C” for every question.

Part 5: Analysis and Reflection (20 pts) 
Provide a discussion (minimum 300 words): 
• Strengths of KNN 
• Limitations of KNN (e.g., computation cost, sensitivity to scaling) 
• When KNN is appropriate or not appropriate 
• Observations from your experiment




Analysis and Reflection
K-Nearest Neighbors (KNN) is a simple yet powerful supervised learning algorithm used for classification and regression. One of the main strengths of KNN is its simplicity. The algorithm is easy to understand and implement because it does not require training a complex model. Instead, it stores all the training data and makes predictions based on the similarity between data points. Another strength is that KNN is non-parametric, meaning it does not assume any underlying distribution of the data. This makes it flexible and effective for real-world datasets such as the diabetes dataset used in this experiment. Additionally, KNN can adapt well to new data, since adding new training examples does not require retraining the model.
Despite these advantages, KNN also has several limitations. One major limitation is computational cost. Since KNN calculates the distance between a test point and all training points, prediction becomes slow when working with large datasets. This makes KNN inefficient compared to algorithms that build compact models. Another limitation is sensitivity to feature scaling. Because KNN relies on distance calculations, features with larger numerical ranges can dominate the distance measure. This is why standardization or normalization is required before applying KNN. KNN is also sensitive to noise and outliers, especially when the value of K is too small. A single noisy data point can significantly affect the prediction.
KNN is most appropriate when the dataset is small to medium-sized and when the relationship between features is complex or unknown. It also works well when similar data points are expected to belong to the same class, such as medical diagnosis datasets. However, KNN is not appropriate for very large datasets due to high computation time. It is also less effective when the dataset contains many irrelevant features or when data is highly imbalanced. In high-dimensional datasets, KNN performance may decrease due to the curse of dimensionality.
From the experiment, it was observed that different values of K produced different predictions. K = 3 correctly classified the test instance, while K = 5 and K = 7 misclassified it. This shows that smaller values of K focus more on nearby neighbors, while larger values consider more distant points that may belong to another class. The results also highlight the importance of selecting an appropriate K value. Additionally, preprocessing steps such as handling missing values and applying feature scaling were necessary to improve model performance. Overall, the experiment demonstrated that KNN is easy to apply and effective, but its performance depends heavily on proper preprocessing and choosing the right value of K.

BONUS(Optional +10pts)
	Visualize results using graphs (e.g., accuracy vs k)
  <img width="975" height="831" alt="image" src="https://github.com/user-attachments/assets/7a4e67e4-f1ee-47d4-9c2d-7dd96752f621" />

The graph shows how the accuracy of the KNN model changes with different values of K (number of nearest neighbors). The x-axis represents K values, while the y-axis shows the model’s accuracy. It helps identify which K value gives the best performance. Generally, accuracy improves as K increases to an optimal point, then may decrease if K becomes too large due to overgeneralization.

	Compare KNN with another algorithm (e.g., Logistic Regression)

K-Nearest Neighbors (KNN) is an instance-based algorithm that classifies a data point based on the majority class of its nearest neighbors. It is simple and effective but can be slow during prediction because it computes distances to all training data. It is also sensitive to feature scaling and noisy data.

Logistic Regression, on the other hand, is a model-based algorithm that learns a mathematical equation to separate classes. It is faster in prediction, works well for linearly separable data, and is less affected by large datasets. However, it may perform poorly if the relationship between variables is complex and non-linear.

# WINE DATASET
<img width="580" height="750" alt="image" src="https://github.com/user-attachments/assets/38a8b7c1-10e5-4a1b-9eba-04ad12c7c265" />
<img width="573" height="776" alt="image" src="https://github.com/user-attachments/assets/4d42a890-397f-4eb9-846d-80fb3174bea5" />
<img width="571" height="752" alt="image" src="https://github.com/user-attachments/assets/97ef8eec-2e42-4ebe-bf4a-b0a4b5a948b0" />
<img width="576" height="767" alt="image" src="https://github.com/user-attachments/assets/66b3a114-ef20-4333-9bac-7a86e79f1d31" />
<img width="574" height="811" alt="image" src="https://github.com/user-attachments/assets/656b65e0-240c-4d32-aee3-6e59d8aa1a95" />
Summary: Wine Dataset using KNN and K-Means
Dataset

The dataset consists of 10 wine samples with two features:

Alcohol
Flavanoids

Each wine is labeled as Class A or Class B. A test wine with values Alcohol = 13.00 and Flavanoids = 2.50 is used for classification.

Part 1: KNN Classification

The Euclidean distance formula was used to compute the distance between the test wine and each wine in the dataset. The computed distances were sorted in ascending order.

Nearest neighbors (k = 3):

Wine8 → Class B
Wine2 → Class A
Wine6 → Class B

Voting:

Class B = 2
Class A = 1

Final KNN Prediction:
Test Wine → Class B

This indicates that the unknown wine is more similar to wines belonging to Class B.

Part 2: K-Means Clustering

Initial centroids were selected:

C1 = Wine1 (14.23, 3.06)
C2 = Wine5 (12.33, 1.95)

Each wine’s distance to both centroids was calculated and assigned to the nearest cluster.

Cluster 1:
Wine1, Wine3, Wine7, Wine9

Cluster 2:
Wine2, Wine4, Wine5, Wine6, Wine8, Wine10

New centroids were computed by averaging values inside each cluster:

New C1:
(13.64 , 3.02)

New C2:
(12.69 , 2.30)

Final Clustering Result

Cluster 1:
Higher alcohol and flavanoids wines
W1, W3, W7, W9

Cluster 2:
Lower alcohol and flavanoids wines
W2, W4, W5, W6, W8, W10

New centroids were computed by averaging values inside each cluster:

New C1:
(13.64 , 3.02)

New C2:
(12.69 , 2.30)

Final Clustering Result

Cluster 1:
Higher alcohol and flavanoids wines
W1, W3, W7, W9

Cluster 2:
Lower alcohol and flavanoids wines
W2, W4, W5, W6, W8, W10

Final Conclusion

KNN Result:
The test wine was classified as Class B based on the majority vote of the nearest neighbors.

K-Means Result:
The dataset was grouped into two clusters. Cluster 1 contained wines with higher alcohol and flavanoids, while Cluster 2 contained wines with lower values.

Overall, KNN was used for classification of the unknown wine, while K-means was used to discover natural groupings within the wine dataset.


