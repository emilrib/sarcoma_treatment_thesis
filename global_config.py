import os

# Base directory
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define subdirectories
analysis_dir = os.path.join(BASE_DIR, "Analysis")
datasets_dir = os.path.join(BASE_DIR, "Datasets")
evaluation_dir = os.path.join(BASE_DIR, "Evaluation")
model_dir = os.path.join(BASE_DIR, "Model")
preprocessing_dir = os.path.join(BASE_DIR, "Preprocessing")
validation_dir = os.path.join(BASE_DIR, "Validation")
# Create folders if they don't exist
for folder in [analysis_dir, datasets_dir, evaluation_dir, model_dir, preprocessing_dir, validation_dir]:
    os.makedirs(folder, exist_ok=True)

#print(f" datasets_dir: {datasets_dir}")


