# Galaxy-Classification-using-SDSS-Data
This project demonstrates a machine learning workflow for classifying galaxies based on their spectral properties using data from the Sloan Digital Sky Survey (SDSS). The project covers data acquisition, preprocessing, model training, and evaluation, using Python and popular data science libraries.

Overview
The program classifies galaxies into two categories based on their redshift, using features such as mean flux and redshift. The main steps include fetching data from the SDSS, preprocessing it, handling class imbalance with SMOTE, training a Random Forest classifier, and evaluating the model's performance.

Features
Data Acquisition:

Queries the SDSS database for spectral data of galaxies in a specified sky region.
Retrieves features like mean flux and redshift for each galaxy.
Data Preprocessing:

Removes missing values from the dataset.
Normalizes features to have zero mean and unit variance.
Creates a binary label (low or high) based on the redshift value.
Feature Extraction and Label Encoding:

Extracts relevant features (flux and redshift) for model input.
Encodes categorical labels into numerical form using LabelEncoder.
Handling Class Imbalance:

Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by generating synthetic samples for the minority class.
Model Training:

Trains a Random Forest classifier on the balanced dataset.
Model Evaluation:

Predicts galaxy classes on the test set.
Evaluates model performance using a classification report and confusion matrix.
Dependencies
Python 3.x
astroquery
astropy
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
Installation
Clone the repository:

sh
Copy code
git clone <repository-url>
cd <repository-directory>
Install required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Run the script:

sh
Copy code
python galaxy_classification.py
The script will query the SDSS, preprocess the data, train the model, and output the classification report and confusion matrix.

Output:

The classification report provides precision, recall, F1-score, and support for each class.
The confusion matrix visualizes the correct and incorrect predictions made by the model.
Project Structure
galaxy_classification.py: The main script that performs all steps from data acquisition to model evaluation.
requirements.txt: A file listing the required Python packages.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
SDSS: We acknowledge the Sloan Digital Sky Survey for providing the astronomical data used in this project.
