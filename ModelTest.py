import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import joblib  # To load the saved model

# Step 1: Feature Extraction for NPK (same as before)
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
    yellowish_blue_ratio = ((mean_color[2] + mean_color[1]) / 2) / mean_color[0]

    # Texture Features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    # GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
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

# Step 2: Predict NPK values from features
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

# Step 3: Test the Model
def test_model_on_image(model, image_path):
    # Extract features from the input image
    features = extract_features(image_path)

    # Predict NPK values based on the extracted features
    npk_values = predict_npk(features)

    # Print the predicted NPK values
    print(f"Predicted NPK values for the image: {npk_values}")

    # If you want to use the model to refine predictions
    # model_prediction = model.predict([list(features.values())])
    # print("Model prediction (Refined):", model_prediction)

# Step 4: Load the trained model and test on an image
if __name__ == "__main__":
    # Load the trained model from the saved file
    model = joblib.load("npk_predictor_model.tflite")

    # Path to the image you want to test
    image_path ="C:/Users/vijay_jjyhjd9/Downloads/WhatsApp Image 2024-11-23 at 20.35.31_d2cbae08.jpg" #Replace with your image path

    # Test the model with the image
    test_model_on_image(model, image_path)
