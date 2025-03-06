# Disaster Classifier

This project classifies disaster-related messages into relevant categories to assist in emergency response. Follow the steps below to set up, train the model, and deploy a Streamlit web application for inference.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sadavaidya/Disaster_classifier.git
cd Disaster_classifier
```

### 2. Create a Conda Virtual Environment

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. Then, create and activate the virtual environment:

```bash
conda create --name disaster_classifier python=3.8
conda activate disaster_classifier
```

### 3. Install Required Packages

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Running the Data Processing Pipeline

Execute the pipeline script to process data and train the model:

```bash
python src/pipeline.py
```

This step will generate the trained model and save it in the `models` directory.

## Running the Streamlit Application for Inference

To deploy the model using Streamlit:

1. Ensure the trained model (`classifier.pkl`) is available in the `models` directory.
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

This command will launch a web interface where users can enter a message and receive a classification.

## Notes
- The Streamlit app (`app.py`) is included in this repository.
- Modify `app.py` as needed to customize the UI or enhance functionality.
- Ensure all dependencies are installed before running the app.

