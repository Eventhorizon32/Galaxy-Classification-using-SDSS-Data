# Import necessary libraries
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Step 1: Data Acquisition
# Define the coordinates for the query
pos = coords.SkyCoord('12h00m00s +00d00m00s', frame='icrs')

# Query SDSS for galaxies in the specified region with a radius less than 3.0 arcmin
xid = SDSS.query_region(pos, radius='2.99arcmin', spectro=True)

# Retrieve the spectra data associated with the galaxies
spectra = SDSS.get_spectra(matches=xid)

# Extracting basic information from spectra
data = []
for spectrum in spectra:
    flux = np.mean(spectrum[1].data['flux'])  # Mean flux as a simple feature
    redshift = spectrum[2].data['Z'][0] if 'Z' in spectrum[2].data.names else np.nan  # Redshift
    data.append({'flux': flux, 'redshift': redshift})

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 2: Data Inspection and Preprocessing
# Display basic information about the dataset
print(df.info())
print(df.describe())

# Handling missing values
df = df.dropna()

# Normalizing numerical features
scaler = StandardScaler()
df[['flux', 'redshift']] = scaler.fit_transform(df[['flux', 'redshift']])

# Create a binary label based on redshift threshold (for demonstration)
df['label'] = df['redshift'].apply(lambda x: 'low' if x < 0 else 'high')  # Assuming binary classification

# Check overall class distribution in the dataset
print("Overall class distribution in the dataset:")
print(df['label'].value_counts())

# Step 3: Feature Extraction and Label Encoding
# Separating features and labels
X = df[['flux', 'redshift']].values  # Convert to numpy array
y = df['label'].values  # Convert to numpy array

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Display encoded labels
print("Encoded labels:", y_encoded[:5])
print("Label classes:", label_encoder.classes_)

# Step 4: Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Ensure `y_train` is a 1-dimensional array
y_train = np.ravel(y_train)

# Display class distribution in the training set
print("Training set class distribution:")
print(pd.Series(y_train).value_counts())

print("Test set class distribution:")
print(pd.Series(y_test).value_counts())

# Step 5: Handling Class Imbalance with SMOTE
# Determine the number of samples for each class
class_counts = pd.Series(y_train).value_counts()
minority_class_count = class_counts.min()

# Set n_neighbors to be one less than the number of samples in the minority class, ensuring it's at least 1
n_neighbors = min(minority_class_count - 1, 5)
if n_neighbors < 1:
    n_neighbors = 1

# Apply SMOTE with adjusted n_neighbors
try:
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=n_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
except ValueError as e:
    print(f"SMOTE could not be applied: {e}")
    X_train_balanced, y_train_balanced = X_train, y_train

# Step 6: Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_balanced, y_train_balanced)

# Step 7: Model Evaluation
# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report with zero_division set to handle undefined metrics
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
