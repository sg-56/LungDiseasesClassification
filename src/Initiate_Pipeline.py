import os
from directory_utils import change_directory, check_directory
from data_visualization import plot_image_distribution, plot_training_history
from model_utils import create_and_train_model, save_model, load_and_evaluate_model

print("Current Working Directory : ",os.getcwd())


# Directory setup
# parent_directory = "../"
# change_directory(parent_directory)

# Dataset setup
input_directory = 'LungXRays-grayscale'
input_directory = os.path.join(os.getcwd(),input_directory)
# a = input()
check_directory(input_directory)

# Define class names
class_names = ["Corona_Virus_Disease", "Normal", "Pneumonia", "Tuberculosis"]
base_path = input_directory

# Data visualization
# plot_image_distribution(base_path, class_names)

# Model creation and training
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
trained_model, training_history = create_and_train_model(train_path, val_path)

#Plot training history
plot_training_history(training_history)

# Save model
model_path = os.path.join(os.getcwd(), "Models")
model_file_path = save_model(trained_model, model_path)

# Load and evaluate model
# model_file_path = r'C:\Users\shiva\OneDrive\Desktop\LungDisease\Models\LungDiseaseModel-CNN.keras'

load_and_evaluate_model(model_file_path, val_path, class_names)



print("Pipeline Complete : Model has been generated and saved!!!")
