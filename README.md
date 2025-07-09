# Wildlife Species Detector

Web application that integrates machine learning models for image classification. This project consists of a Flask backend and a React frontend.

This is a computer vision project aimed to identify endangered species and rare animal sightings.

## Dataset
Our data set is from [LILA BC](https://lila.science/datasets/nacti). It contains 3.7M camera trap images from five locations across the United States, with labels for 28 animal categories, primarily at the species level (for example, the most common labels are cattle, boar, and red deer).

## Preprocessing
Starting with such a large dataset, we narrowed it down to around 70k images.

## Model Training
* Randomly sampled 2% of the full dataset (67,000 out of 3.2 million images), preserving the original class distribution in the subset.
* Preprocessing pipeline included resizing images to 224×224 pixels, followed by feature extraction using ResNet50, producing a 2048-dimensional feature vector for each image.
* For data sampling and evaluation:
    * Allocated 20% of the 2% subset for testing.
    * Both Logistic Regression and SVM achieved approximately 84% accuracy.
    * SVM showed slightly better performance than Logistic Regression, so it was chosen for further fine-tuning.
* For kernel selection during SVM fine-tuning:
    * Analyzed the confusion matrix and overall classification performance.
    * Observed strong linear separability in the dataset.
    * Chose a linear kernel for the SVM, as it is well-suited for high-dimensional, linearly separable data.

## Prerequisites

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





