# Logistic Regression on the Iris Dataset

This project implements a logistic regression model to classify the Iris dataset. The Iris dataset is a classic machine learning dataset containing measurements of iris flowers and their corresponding species. This notebook demonstrates how to preprocess the data, train a logistic regression model, and evaluate its performance.

## Features of the Iris Dataset

The dataset contains 150 samples with the following features:

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

Each sample is labeled as one of three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Project Workflow

1. **Data Loading**: Import the Iris dataset from `sklearn.datasets`.
2. **Exploratory Data Analysis (EDA)**: Visualize the dataset to understand patterns and relationships.
3. **Data Preprocessing**: Standardize the dataset to improve model performance.
4. **Model Training**: Train a logistic regression model using `scikit-learn`.
5. **Evaluation**: Measure the model’s accuracy using metrics like confusion matrix, precision, recall, and F1 score.
6. **Visualization**: Plot decision boundaries to visualize model performance.

## Requirements

The following Python libraries are required to run the notebook:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone this repository or download the notebook file.
2. Ensure you have Python (3.7 or higher) installed on your machine.
3. Install the required dependencies listed above.
4. Open the notebook in Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook logistic_regression_iris_dataset.ipynb
   ```

5. Run each cell in sequence to reproduce the results.

## Results

The logistic regression model achieves high accuracy on the Iris dataset, demonstrating its effectiveness in multi-class classification problems. Decision boundary visualizations provide insights into how the model separates different species.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The Iris dataset is provided by the UCI Machine Learning Repository.
- Thanks to the `scikit-learn` library for its tools and documentation.
