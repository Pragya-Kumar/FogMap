# 🌫️ Fog vs. Clear Image Classifier (Deep Learning)

## 📌 Project Overview
This project is a Deep Learning computer vision model designed to classify images into two categories: **Clear** or **Smog/Fog**. 

Built using TensorFlow and Keras, this project demonstrates the complete machine learning lifecycle: data cleaning, exploratory data analysis (EDA), transfer learning, and fine-tuning to overcome dataset bias (Domain Shift).

## 📊 The Dataset
The model was initially trained on the **SMOG4000** dataset, which consists of 2,560 perfectly balanced images:
* **1280 Clear Images:** Highways, clear cityscapes, and traffic.
* **1280 Smog Images:** Heavily polluted, grey/brown urban environments.

*Note: The dataset and saved model files are excluded from this repository due to size constraints.*

## 🧠 Methodology & Architecture
1. **Data Preprocessing:** Images were resized to `224x224` and normalized.
2. **Data Augmentation:** Applied dynamic random flipping, rotation, and zooming during training to prevent overfitting.
3. **Transfer Learning (Feature Extraction):** Utilized the pre-trained **MobileNetV2** architecture (frozen base) to extract spatial features and edge geometries. Added a custom GlobalAveragePooling2D and Dense Sigmoid classification head.
4. **Fine-Tuning:** Unfroze the top 54 layers of MobileNetV2 with a micro-learning rate (0.00001) to allow the model to learn the specific mathematical pixel gradient of "haze".

## 🚀 Results & Overcoming Dataset Bias
### Initial Training (Base Model)
* **Test Accuracy:** 97.00%
* **Test Recall:** 97.00%
* **The Blind Spot:** Despite high accuracy, the model failed on Out-of-Distribution (OOD) data. When tested on a picture of a misty, white forest road, it predicted "Clear" (71% confidence) because it had only ever been trained on *grey urban smog*. It relied too heavily on high-contrast road lines and dark trees.

### Fine-Tuning (The Fix)
By fine-tuning the deep feature-extraction layers, the model learned the actual definition of atmospheric opacity rather than memorizing city pollution.
* **Fine-Tuned Validation Accuracy:** 98.91%
* **OOD Resolution:** The fine-tuned model successfully predicted the previously failed misty forest image as **"Smog" with 96.1% confidence**, and improved its confidence on clear highways to **99.8%**.

## 🛠️ Technologies Used
* Python 3.x
* TensorFlow / Keras
* Scikit-Learn (Classification reports & Confusion Matrices)
* Matplotlib & Seaborn (Visualizations)
