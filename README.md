# Wildlife Species Detector

Web application that integrates machine learning models for image classification. This project consists of a Flask backend and a React frontend.

## Dataset
Our data set is from [LILA BC](https://lila.science/datasets/nacti). It contains 3.7M camera trap images from five locations across the United States, with labels for 28 animal categories, primarily at the species level (for example, the most common labels are cattle, boar, and red deer).

## Preprocessing
We narrowed our dataset down to around 70k images. To prepare the dataset for training, we followed this cleaning and balancing process using the metadata in `nacti_metadata.csv`:

**1. Removed Missing Labels:**
* We dropped all rows with missing values to ensure clean and usable data for modeling.

**2. Created Visualizations:**
* We visualized the distribution of image labels (`common_name`) using a horizontal bar chart, which showed major class imbalances — some species had thousands of samples while others had very few.

**3. Class Balancing (At Species Level):**
* To reduce bias, we performed class balancing between "red deer" and "domestic cow" by:
   * Sampling the same number of "domestic cow" instances as there were "red deer"
   * Removing all original "domestic cow" entries
   * Combining the balanced sample back into the dataset
  
**4. Class Balancing (At Order Level):**
* We balanced higher-level taxonomic orders by:
   * Sampling the same number of "artiodactyla" as there were "carnivora"
   * Dropping the original "artiodactyla" rows
   * Adding the balanced data with the rest of the dataset

**5. Frequency Filtering and Truncation:**
* To remove underrepresented classes and reduce memory usage:
   * We dropped all species (`common_name`) with fewer than 100 occurrences
   * We capped the maximum number of samples per species to 40,000

**6. Final Dataset Prep:**
* After filtering and balancing:
   * Each species had at least 100 examples, but no more than 40,000
   * The dataset was reset and ready for feature extraction + model training

**7. Label Encoding and Data Splitting:**
   * Species labels were encoded into numeric values using `LabelEncoder`
   * The feature matrix `X` and label vector `y` were split into training/validation and test sets (80/20 split) using `train_test_split`

This preprocessing pipeline helped reduce class imbalance and ensured the dataset was clean, representative, and well-structured for model evaluation.

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
 
## Challenges
* Searching for an adequate dataset
* Figuring out what parameters we need to train our model on - ResNet50 or ResNet18
* Reducing the size of our dataset, while preserving its integrity
   * 2.8 million data points after filtering 46 classes
* Improving the accuracy of our model


## Try It Out

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





