# Wildlife Species Image Classification Project 

Web application that integrates machine learning models for image classification. This project consists of a Flask backend to serve ML models and a React frontend for the user interface.

---

## Prerequisites

Make sure the following are installed:

- Python 3.8 or higher 
- Node.js and npm
- Git

---

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install flask
pip install flask-cors
pip install numpy
pip install Pillow
pip install joblib
pip install tensorflow
```
4. Download Model Files
* The model files (.pkl) are too large to be included in the repository
* Download the following files from the shared drive and place them in the backend/ directory:

svm_model.pkl

logistic_regression_model.pkl

label_encoder.pkl

scaler.pkl

Start the Flask server:
bash
Copy
Edit
python app.py
The server will run on http://localhost:5000

3. Frontend Setup
Open a new terminal and return to the root directory:
bash
Copy
Edit
cd ../
Install frontend dependencies:
bash
Copy
Edit
npm install
Start the development server:
bash
Copy
Edit
npm run dev
The frontend will be available at http://localhost:5173

📁 Project Structure
css
Copy
Edit
team_won_website/
├── backend/
│   ├── app.py
│   ├── svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── frontend/
│   └── src/
│       ├── components/
│       └── ...
📸 Usage
Ensure both backend and frontend servers are running.

Open your browser and navigate to http://localhost:5173

Upload an image via the web interface.

The model will process the image and display the prediction result.

🛠️ Troubleshooting
Ensure all .pkl model files are correctly downloaded and placed in the backend/ directory.

Confirm that both the backend and frontend servers are running simultaneously.

Check the browser or terminal console for error messages if something isn’t working.

📬 Contact
For questions or feedback, please reach out via GitHub Issues or contact the contributors directly.

