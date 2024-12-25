# Rock Vs Mine Prediction

This project demonstrates a binary classification problem where the task is to predict whether a given data sample corresponds to a **rock** or a **mine** based on sonar signal data. The implementation is performed in Python using the **Logistic Regression** model from the **scikit-learn** library.

## Dataset

The dataset used in this project is the **Sonar Data**. It consists of 60 features representing sonar signals and a target label:
- **R**: Represents a rock.
- **M**: Represents a mine.

### Dataset Characteristics:
- The dataset is loaded from a `.csv` file without a header row.
- **Shape**: 208 rows and 61 columns (60 features and 1 target label).

## Dependencies
The following Python libraries are required to run the code:
- `numpy`
- `pandas`
- `scikit-learn`

## Steps in the Project

### 1. Data Loading and Preprocessing
- Load the dataset using `pandas.read_csv()`.
- Check the data's structure and look for missing values.
- Separate the features (`X`) and the target labels (`Y`).

### 2. Data Splitting
- Split the dataset into training and test sets using `train_test_split()` from scikit-learn.
- **Stratify** is used to ensure equal distribution of classes in both sets.
- Test size: 10% of the dataset.

### 3. Model Training
- A Logistic Regression model is used for binary classification.
- Train the model using the `fit()` function on the training data.

### 4. Model Evaluation
- Measure the accuracy of the model on both training and test sets using `accuracy_score()`.

### 5. Prediction System
- Input data is provided as a tuple.
- The input data is reshaped into a 2D array before making predictions.
- The model predicts whether the sample corresponds to a **rock** or a **mine**.

## Results
- **Training Accuracy**: Displays the model's accuracy on the training dataset.
- **Test Accuracy**: Displays the model's accuracy on the test dataset.
- **Prediction System**: Outputs whether the input data corresponds to a rock or a mine.

## Usage
### Running the Code
1. Place the `sonar data.csv` file in the appropriate path (`/content/drive/MyDrive/datasets/`).
2. Execute the script in a Jupyter Notebook or Google Colab environment.
3. Provide sample input data to test the predictive system.

### Example
#### Input Data
```
(0.0093, 0.0185, 0.0056, 0.0064, 0.0260, 0.0458, 0.0470, 0.0057, 0.0425, 0.0640, ... , 0.0025)
```
#### Output
```
M  -> It is a Mine...!!
R  -> It is a Rock...!!
```

## Notes
- Logistic Regression is suitable for binary classification problems like this.
- The dataset should be preprocessed to ensure optimal performance.
- Accuracy may vary depending on hyperparameters and dataset splitting.

## Improvements
- Consider exploring other machine learning models like Decision Trees or Support Vector Machines to compare performance.
- Perform feature scaling to improve model performance.
- Use cross-validation for better evaluation.

## License
This project is distributed under the MIT License.
