# 🤖 AI Engineering Portfolio Hub

### A Unified Showcase of Machine Learning, Deep Learning, and Computer Vision Projects

[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

-----

## 🎯 Introduction and Goal

This repository serves as a unified **AI Engineering Portfolio**, designed to demonstrate solid expertise in Machine Learning, Deep Learning, Computer Vision, Python Backend Development, and MLOps.

Each project is hosted within a single, multi-page **Streamlit** application, allowing for a seamless and functional demonstration of models and analyses, which is crucial for interviews and professional opportunities. The modular structure ensures **scalability** and **maintainability** .

-----

## 📂 Project Structure

The repository follows a clear, modular architecture, separating presentation, core business logic, and models.

```
AI-Project/
├── app.py                      # Main Streamlit entry point (Home Page)
├── pages/
│   ├── 1_🤖_Fake_vs_Real_Classifier.py  # Scripts for Streamlit side pages (your projects)
│   ├── 2_🎭_Face_Morphing.py
│   └── ...
├── src/                        # Python modules with core logic (ML, analysis, utilities)
│   ├── ml_inference.py         # General inference logic
│   ├── adult_logic.py          # Specific logic for Adult Census
│   └── ...
├── models/                     # Pre-trained models and configuration files
│   ├── cnn/                    # Deep Learning models (.keras)
│   ├── cv/                    # Computer Vision assets (.xml, .dat)
│   └── lbp/                    # Traditional ML models (.pkl)
├── data/                       # Data used for analysis or demos
├── Chess/                      # Isolated module for the Chess Engine (Pygame)
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

-----

## ✨ Contained Projects

Below is a list of the AI and Data Science mini-projects included in the portfolio.

| Icon | Project Name | AI Type/Technology | Brief Description |
| :---: | :--- | :--- | :--- |
| 🤖 | **Fake vs Real Classifier** | Binary Classification/CV | A classifier to distinguish between real and generated/manipulated images using LBP features and a traditional ML model. |
| 🎭 | **Face Morphing** | Computer Vision (CV) | Demonstration of image manipulation and transition between faces using advanced CV techniques (dlib, OpenCV). |
| 🧠 | **DFS Algorithmic Solvers** | Search Algorithms | Implementation and visualization of solutions to classic problems (e.g., N-Queens, Maze) using the Depth First Search (DFS) algorithm. |
| ♟️ | **Pygame Chess Engine** | Game AI/External Engine | Integration of a chess engine (Stockfish) within a Pygame interface for analysis and gameplay. (Note: Run separately). |
| 📊 | **Clustering Analysis** | Unsupervised Machine Learning | Execution and visualization of clustering algorithms (e.g., K-Means) for unlabeled data analysis. |
| 📈 | **Keras Regression** | Deep Learning (TensorFlow) | A neural network model (CNN) for regression tasks, focused on optimization and training. |
| 🖼️ | **MNIST Classification** | Deep Learning (CNN) | Handwritten digit recognition using a Convolutional Neural Network (CNN) on the MNIST dataset. |
| 🚢 | **Titanic Survival** | Classification (Traditional ML) | Prediction of Titanic passenger survival using feature engineering and a classification model (e.g., XGBoost). |
| 🏠 | **Housing Regression** | Regression (Traditional ML) | A predictive model to estimate house prices based on complex datasets. |
| 💰 | **Adult Census** | Classification (Traditional ML) | Income prediction (\>50k or \<=50k) using advanced preprocessing techniques for categorical data and XGBoost. |
| 🏦 | **Bank Marketing** | Classification (Imbalanced Learning) | Prediction model for subscription to a banking product, with emphasis on managing imbalanced datasets (`imbalanced-learn`). |
| 🗺️ | **California Housing Analysis** | Data Analysis & Viz | Comprehensive data exploration and statistical analysis of the California Housing dataset, with interactive visualizations. |

-----

## 🛠️ Key Technologies and Libraries

The project is built entirely in Python and leverages the following libraries, listed in `requirements.txt`:

### 🚀 MLOps & Frontend Stack

  * **`streamlit`**: Framework used to create the web interface and unify all projects into a single interactive dashboard.

### 🧠 Machine Learning & Deep Learning

  * **`tensorflow`**: For Deep Learning projects (Keras Regression, MNIST Classification).
  * **`scikit-learn`**: For classical Machine Learning algorithms (Classification, Regression, Clustering, Preprocessing).
  * **`xgboost`**: High-performance Boosting model, used for complex classification problems (Titanic, Adult Census, Bank Marketing).
  * **`imbalanced-learn`**: Essential toolkit for addressing imbalanced dataset issues (Bank Marketing).
  * **`category_encoders`**: For advanced encoding of categorical variables in ML pipelines.

### 🖼️ Computer Vision & Multimedia

  * **`opencv-python-headless`**: Fundamental library for image processing and Computer Vision projects (Face Morphing, Fake vs Real Classifier).
  * **`imageio`**: For reading and writing multimedia files.

### 📈 Data Science & Utility

  * **`numpy`** / **`pandas`**: Efficient data manipulation and analysis.
  * **`seaborn`** / **`matplotlib`**: Creation of static and statistical data visualizations.
  * **`requests`**: For HTTP/API interactions, if required in certain modules.

-----

## ⚙️ How to Run the Project (Locally)

Follow these steps to launch the entire suite of projects locally.

### 1\. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-Project
```

### 2\. Set up the Python Environment

It is highly recommended to use a virtual environment (`venv` or `conda`).

```bash
# Create and activate the environment (with venv)
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

### 3\. Install Dependencies

Install all necessary libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4\. Run the Streamlit Application

Launch the main application. Streamlit will automatically load all pages present in the `pages/` folder.

```bash
streamlit run app.py
```

The application will be available in your browser, typically at `http://localhost:8501`. All projects will be accessible through the sidebar navigation.

-----

## ⚠️ Caveats and Important Notes

### Additional Dependencies for Specific Modules

  * **Face Morphing:** The Face Morphing project (`2_🎭_Face_Morphing.py`) requires the `shape_predictor_68_face_landmarks.dat` file and potentially the **`dlib`** library, which is known for its complex compilation dependencies. If you encounter issues, you might need to install `dlib` separately or from a pre-compiled wheel.
  * **Chess Engine:** The Chess project (`4_♟️_Motore_Scacchistico_Pygame.py`) is a standalone **Pygame** application that does not integrate directly into the Streamlit interface. It resides in a separate Python module (`Chess/`) which will need to be executed independently to function correctly.
  * **Pre-trained Models:** Certain models, such as `modello_cnn.keras` or `.pkl` files, are included in the `models/` folder. These files can be large. In a real MLOps environment, these would typically be managed via a model versioning system (e.g., MLflow) or cloud storage (e.g., S3/GCS).
