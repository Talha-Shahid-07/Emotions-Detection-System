# Emotions Detection System

A machine learning project developed for the Programming for Artificial Intelligence course at Bahria University Karachi Campus. This system detects human emotions from facial landmark data using fuzzy logic and a Random Forest classifier. It processes the RAVDESS dataset, extracts facial features (e.g., eye openness, mouth width, Action Units), and predicts emotions such as neutral, happy, sad, surprised, and angry. The project includes data preprocessing, fuzzy logic modeling, and real-time webcam data analysis.

## 🌟 Features
- **Data Preprocessing**: Cleans and normalizes facial landmark data from the RAVDESS dataset, handling outliers and missing values.
- **Feature Extraction**: Computes features like eye openness, mouth width, and Action Units (AU06, AU12, AU04, etc.) for emotion analysis.
- **Fuzzy Logic System**: Uses `skfuzzy` to classify emotions based on facial features, with rules for emotions like happy, sad, surprised, and neutral.
- **Random Forest Classifier**: Trains a balanced Random Forest model to predict emotions, with evaluation via classification report and confusion matrix.
- **Real-Time Webcam Analysis**: Processes OpenFace-generated CSV data from webcam input to predict emotions frame-by-frame.
- **Visualization**: Generates a confusion matrix heatmap to evaluate model performance.

## 🛠️ Technologies Used
- **Programming Language**: Python 3.8+
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and preprocessing
  - `scikit-fuzzy`: Fuzzy logic system for emotion classification
  - `scikit-learn`: Random Forest classifier and evaluation metrics
  - `matplotlib`, `seaborn`: Visualization of confusion matrix
  - `joblib`: Model and scaler persistence
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Tools**: Google Colab, OpenFace (for webcam data processing)

## 🚀 Setup and Installation

### Prerequisites
- **Python**: Version 3.8 or higher.
- **Google Colab**: Recommended for running the notebook, as the code was developed in Colab.
- **RAVDESS Dataset**: Download the dataset and place `archive.zip` in the project directory or Google Drive.
- **OpenFace**: Required for generating webcam facial landmark data (optional for real-time analysis).

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Emotions-Detection-System.git
   cd Emotions-Detection-System
   ```
2. **Install Dependencies**:
   Install required Python packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset**:
   - Place `archive.zip` (RAVDESS dataset) in the project root or Google Drive (`/content/drive/MyDrive/AI_Emotion_Project/`).
   - Update the `zip_path` in `emotions_detection_system_final_2_0.py` to match your dataset location.
4. **Run the Code**:
   - **Jupyter Notebook**: Open `emotions_detection_system_final_2_0.ipynb` in Jupyter or Colab and execute cells sequentially.
   - **Python Script**: Run `emotions_detection_system_final_2_0.py` directly if using a local environment with dependencies installed:
     ```bash
     python emotions_detection_system_final_2_0.py
     ```
5. **Webcam Analysis** (Optional):
   - Generate facial landmark data using OpenFace and save as `surprised sad confused crying.csv`.
   - Update the file path in the script to process webcam data.

## 💡 Usage
- **Preprocessing**: The script extracts and cleans data from `archive.zip`, producing `normalized_features_with_labels.csv`.
- **Fuzzy Logic**: Applies fuzzy rules to classify emotions, saving results to `fuzzy_emotion_scores_final.csv`.
- **Random