import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # To save and load the model

# Step 1: Feature Extraction for NPK


def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Compute mean and std for RGB channels
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))

    # Compute normalized RGB
    sum_rgb = np.sum(mean_color)
    norm_r = mean_color[2] / sum_rgb
    norm_g = mean_color[1] / sum_rgb
    norm_b = mean_color[0] / sum_rgb

    # Nitrogen indicator (G dominance)
    green_dominance = mean_color[1] > (mean_color[0] + mean_color[2]) / 2

    # Phosphorus indicator (Bluish soil)
    blue_ratio = mean_color[0] / (mean_color[1] + mean_color[2])

    # Potassium: Yellowish-Blue Ratio
    yellowish_blue_ratio = (
        (mean_color[2] + mean_color[1]) / 2) / mean_color[0]

    # Texture Features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 10), density=True)

    # GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1], angles=[
                        0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    # Combine features
    features = {
        "mean_R": mean_color[2],
        "mean_G": mean_color[1],
        "mean_B": mean_color[0],
        "std_R": std_color[2],
        "std_G": std_color[1],
        "std_B": std_color[0],
        "norm_R": norm_r,
        "norm_G": norm_g,
        "norm_B": norm_b,
        "green_dominance": green_dominance,
        "blue_ratio": blue_ratio,
        "yellowish_blue_ratio": yellowish_blue_ratio,
        "lbp_hist": lbp_hist.tolist(),
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_entropy": glcm_entropy,
    }
    return features

# Step 2: Map Features to NPK Values


def predict_npk(features):
    # Enhanced feature-based prediction for NPK values

    # Nitrogen Prediction (Green dominance, more complex rule)
    if features["green_dominance"]:
        nitrogen = 75  # High nitrogen due to significant green
    elif features["mean_G"] > (features["mean_R"] + features["mean_B"]) / 2:
        nitrogen = 60  # Moderate nitrogen based on a threshold
    else:
        nitrogen = 40  # Low nitrogen

    # Phosphorus Prediction (Blue ratio)
    if features["blue_ratio"] > 0.8:
        phosphorus = 70  # High phosphorus due to bluish tint
    elif features["blue_ratio"] > 0.5:
        phosphorus = 50  # Moderate phosphorus
    else:
        phosphorus = 30  # Low phosphorus

    # Potassium Prediction (Yellowish-blue ratio)
    if features["yellowish_blue_ratio"] > 1.5:
        potassium = 60  # High potassium (due to more yellowish-blue)
    elif features["yellowish_blue_ratio"] > 1.0:
        potassium = 40  # Moderate potassium
    else:
        potassium = 20  # Low potassium

    return {"Nitrogen": nitrogen, "Phosphorus": phosphorus, "Potassium": potassium}

# Step 3: Process Dataset


def process_dataset(folder_path):
    feature_data = []

    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a folder
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file.endswith((".jpg", ".png", ".jpeg")):
                    # Extract features
                    features = extract_features(file_path)
                    npk_values = predict_npk(features)
                    features.update(npk_values)  # Add NPK to feature vector
                    features["soil_type"] = subfolder  # Add soil type as label
                    feature_data.append(features)

    return pd.DataFrame(feature_data)

# Step 4: Train a Model (Optional)


def train_model(train_df):
    # Prepare feature matrix and target labels
    X = train_df.drop(
        columns=["Nitrogen", "Phosphorus", "Potassium", "soil_type", "lbp_hist"])
    y = train_df[["Nitrogen", "Phosphorus", "Potassium"]]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Validate the model
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val , predictions)
    print("Validation MSE:", mse)

    return model


# Step 5: Automate the Process
if __name__ == "__main__":
    train_dir = "Dataset/Trains"
    test_dir = "Dataset/Tests"

    # Process training and testing datasets
    train_df = process_dataset(train_dir)
    test_df = process_dataset(test_dir)

    # Save processed data
    train_df.to_csv("train_features.csv", index=False)
    test_df.to_csv("test_features.csv", index=False)

    # Train and validate the model
    print("Training the model...")
    model = train_model(train_df)

    # Save the model
    joblib.dump(model, "npk_predictor_model.tflite")
    print("Model saved as 'npk_predictor_model.tflite'")

    # Save the feature extraction process
    print("Feature extraction and training completed!")