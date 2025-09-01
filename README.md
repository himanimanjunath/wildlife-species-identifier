# Wildlife Species Detector

Web application that integrates machine learning models for image classification. This project consists of a Flask backend and a React frontend.

<img width="1642" height="779" alt="12" src="https://github.com/user-attachments/assets/cbcabe2b-569c-422d-8880-c4973d0e487f" />


## Timeline
February 2025 - June 2025

## About Machine Learning Student Network
Machine Learning Student Network (MLSN) at UC Davis fosters a collaborative environment where students can build skills, learn, and grow in the field of machine learning.

## Context
As part of the spring cohort, a team of four students and I developed a computer vision project to assist scientists in identifying endangered species and rare animal sightings. Our goal was to support conservationists in monitoring animal populations and detecting species in unexpected environments, which can be crucial for ecological research and conservation efforts. 

## Dataset
Our data set is from [LILA BC](https://lila.science/datasets/nacti). It contains 3.7M camera trap images from five locations across the United States, with labels for 28 animal categories, primarily at the species level (for example, the most common labels are cattle, boar, and red deer).

## Preprocessing
First we cleaned and balanced our dataset using the metadata in `nacti_metadata.csv`, narrowing it down to ~70k images:

**1) Removed missing labels:** Dropped all rows with missing values.

**2) Created visualizations:** Visualized the distribution of image labels (`common_name`) using a horizontal bar chart, which showed major class imbalances; some species had thousands of samples while others had very few.

**3) Balanced classes (at the species level):** To reduce bias, we performed class balancing between "red deer" and "domestic cow" by:
* Sampling the same number of "domestic cow" instances as there were "red deer"
* Removing all original "domestic cow" entries
* Combining the balanced sample back into the dataset
  
**4) Balanced classes (at the order level):** We balanced higher-level taxonomic orders by:
* Sampling the same number of "artiodactyla" as there were "carnivora"
* Dropping the original "artiodactyla" rows
* Adding the balanced data with the rest of the dataset

**5) Filtered and truncated:** To remove underrepresented classes and reduce memory usage:
* Dropped all species (`common_name`) with fewer than 100 occurrences
* Capped the maximum number of samples per species to 40,000

Initial sampling: From the full dataset of 3.7M images, we first randomly sampled 2% (~67k images), preserving the original class distribution. All subsequent preprocessing steps were performed on this 67k subset.

**6) Encoded labels and split data:**
* Encoded species labels into numeric values using `LabelEncoder`
* The feature matrix `X` and label vector `y` were split into training/validation and test sets (80/20 split) using `train_test_split`

This preprocessing pipeline reduced class imbalance and ensured the dataset was clean and representative for model evaluation. 

<img width="1915" height="566" alt="7" src="https://github.com/user-attachments/assets/c1679550-a127-4bbc-be56-fbf53ec280dc" />

<img width="1920" height="818" alt="8" src="https://github.com/user-attachments/assets/18e6820d-91b2-4c19-9e11-d1cd74027ae7" />

## Model Training
We trained models on the preprocessed subset (~67k images). Images were resized to 224×224 pixels, followed by feature extraction using ResNet50, producing a 2048-dimensional feature vector for each image.

* For data sampling and evaluation:
    * Allocated 20% of the 2% subset for testing.
    * Both Logistic Regression and SVM achieved approximately 84% accuracy.
    * SVM showed slightly better performance than Logistic Regression, so it was chosen for further fine-tuning.
* For kernel selection during SVM fine-tuning:
    * Analyzed the confusion matrix and overall classification performance.
    * Observed strong linear separability in the dataset.
    * Chose a linear kernel for the SVM, as it is well-suited for high-dimensional, linearly separable data.
 
<img width="1920" height="895" alt="10" src="https://github.com/user-attachments/assets/cb718b94-04cf-4396-ad8e-acfc72ea2252" />

<img width="1913" height="926" alt="11" src="https://github.com/user-attachments/assets/cd16b6ef-890c-488d-a982-57bb08db006c" />

## Deployment
### Frontend
I took on the responsibility of designing and implementing the frontend for this project to showcase our work at the end-of-quarter demo. Using React (bundled with Vite) and custom CSS, I built an interface within 24 hours that allows users to upload an image and receive a species prediction.

<img width="1396" height="678" alt="Screenshot_2025-05-28_at_5 37 29_PM" src="https://github.com/user-attachments/assets/11a3ebba-cedb-4fcb-aad9-8c8c162ed7a7" />

<img width="1390" height="697" alt="Screenshot_2025-05-28_at_5 37 47_PM" src="https://github.com/user-attachments/assets/32d4f912-fad1-456f-a3ac-7d3e62bd695f" />

Features
* Upload wildlife images directly from your device
* Display predicted species name 
* Mocked prediction logic to simulate ML results, to show the interface is functional while our backend integration was in progress

For ML Integration:
I had already structured the frontend so that it would seamlessly connect with the Flask backend when it was ready. Once deployed, the handleImageUpload function in App.jsx would forward uploaded images to the backend API, which will process them through the ResNet50 + SVM pipeline and return real prediction results.

### Backend
The backend was developed using Flask, serving as the bridge between our trained machine learning model and React frontend. Its primary role is to receive image uploads, preprocess them, run inference, and return predictions in a structured format. The backend hosts the ResNet50 feature extractor and the SVM classifier, which together generate species predictions with confidence scores. It exposes REST API endpoints (e.g., /predict) that can be called directly by the frontend.

Features:
* Accepts image uploads from the frontend
* Preprocesses images (resizing, normalization) before feature extraction
* Runs inference with ResNet50 + SVM and returns species predictions with confidence
* Returns responses in JSON format for integration with React

## Project Tech Stack
### Data & preprocessing
* Python (Pandas, NumPy) → data cleaning, filtering, class balancing, label encoding.
* Matplotlib / Seaborn → visualizations (distribution plots, bar charts for class balance).
* Scikit-learn (LabelEncoder, train_test_split, LogisticRegression, SVM) → label encoding, dataset splitting, baseline classifiers, evaluation.

### Feature extraction & model training
* TensorFlow / Keras (ResNet50 pretrained model) → extracted 2048-dimensional feature vectors from images.
* Scikit-learn (SVM, Logistic Regression, confusion matrix, accuracy metrics) → model training, fine-tuning, evaluation.
* OpenCV / PIL (Python Imaging Library) → resizing and preprocessing images to 224×224.

### Backend
* Flask → REST API server to handle requests:
  * /predict endpoint received uploaded images from the frontend.
  * Preprocessed the image, ran it through ResNet50 → feature vector → SVM → predicted species.
  * Returned prediction as JSON.


### Frontend
* React.js → built the user-facing web app:
  * Upload component for submitting images.
  * Displaying predicted animal species returned by Flask.
  * Basic UI for interacting with the model.
  * Axios / Fetch API → sending POST requests with images to Flask backend and receiving responses.
  * HTML/CSS → styling and layout.


## Challenges
* Searching for an adequate dataset
* Figuring out what parameters we need to train our model on - ResNet50 or ResNet18
* Reducing the size of our dataset, while preserving its integrity
   * 2.8 million data points after filtering 46 classes
* Improving the accuracy of our model


## Try It Out!

### Requirements
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Node.js and npm
- Git

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/pranauww/wildlife-species-identifier.git
cd team_won_website
```

### 2. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended but not required):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install flask
pip install flask-cors
pip install numpy
pip install Pillow
pip install joblib
pip install tensorflow
```

4. Download Model Files:
   - The model files (.pkl) are too large to be included in the repository
   - Download the following [trained models](https://drive.google.com/drive/u/0/folders/14Trgbjr6yKJC4dIYobDeo1w65l0gDyEi) from our shared drive.
     - svm_model.pkl
     - logistic_regression_model.pkl
     - label_encoder.pkl
     - scaler.pkl
   - Place these files in the `backend` directory

5. Start the Flask server:
```bash
python app.py
```
The server will run on `http://localhost:5000`

### 3. Frontend Setup

1. Open a new terminal and navigate to the project root directory

2. Install frontend dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```
The website will be available at `http://localhost:5173`

## Project Structure

```
team_won_website/
├── backend/
│   ├── app.py
│   ├── svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── frontend/
│   ├── src/
│       ├── components/
│       └── ...
```

## Usage

1. Ensure both the backend and frontend servers are running
2. Open your browser and navigate to `http://localhost:5173`
3. Upload an image through the web interface
4. The model will process the image and return the prediction

## Troubleshooting

- If you encounter any issues with the model files, ensure they are properly downloaded and placed in the backend directory
- Make sure both the backend and frontend servers are running simultaneously
- Check the console for any error messages





