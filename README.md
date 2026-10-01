# AI Based Multi-Class Ensemble Learning System for Cervical Cancer Detection

## About

This project focuses on detecting cervical cancer using a deep learning ensemble architecture. The system classifies cervical cytology images into:

1.Normal

2.Precancerous

3.Malignant


The solution integrates EfficientNetB0 and MobileNetV2 (fine-tuned), combined using soft probability voting, achieving a final accuracy of 95%.
A Flask-based web app enables users to upload images and receive predictions, along with a downloadable medical-style PDF report.

## 🎥 Project Demo
Watch the complete project demonstration on YouTube through the below attached link:

https://youtu.be/C4B1-b9jOIo?si=Gl8sl2LTzwg2NLVb

## Features

📤 Upload cervical cell image

🤖 Real-time prediction using ensemble deep learning

📊 Confidence score for each class

📝 Enter patient details before prediction

📄 Downloadable PDF medical report

## Development Requirements

<img width="657" height="317" alt="1" src="https://github.com/user-attachments/assets/ee24ef97-3d2a-4524-9c1b-45c3d2f4f603" />

## System Architecture

![2](https://github.com/user-attachments/assets/9e383925-7662-4de2-a561-c26713d4da8c)

📌 Ensemble Formula:
final_probabilities = (EfficientNetB0_output + MobileNetV2_output) / 2

## Methodology
### 1. Data Preprocessing

i) The images from the Mendeley Cervical Cytology Dataset were cleaned by removing corrupted or unreadable files.

ii) All images were resized to 224 × 224 px, normalized, and converted into a consistent RGB format suitable for CNN processing.

iii) Data augmentation techniques such as rotation, zoom, brightness shift, and horizontal flip were applied to improve generalization and reduce overfitting.

### 2. Model Training

i) Two deep learning models were used for feature extraction and classification:

1.EfficientNetB0 (Pretrained on ImageNet)

2.Fine-Tuned MobileNetV2

ii) The outputs of both models were combined using a soft probability voting ensemble technique, forming a final deployable model named:
ensemble_model.keras

iii) The model was trained in Google Colab using GPU acceleration with Adam optimizer, categorical cross-entropy loss, and early stopping to prevent overfitting.

### 3. Model Evaluation

Evaluation metrics included: accuracy, precision, recall, F1-score, and confusion matrix.

The individual models and the ensemble model were evaluated using accuracy, precision, recall, F1-score, and confusion matrix to compare their classification performance.

The final deployed model achieved:

<img width="507" height="346" alt="3" src="https://github.com/user-attachments/assets/234f3fdb-1eb5-4ddc-8efc-bc6483291d30" />

<img width="676" height="215" alt="4" src="https://github.com/user-attachments/assets/a06fd009-e392-47c7-8b8c-14e7731f9441" />

### 4. Setup Instructions
#### Run the Flask Web App:
```
.\venv\Scripts\Activate
python app.py
```
#### Access Web Interface:
```
http://127.0.0.1:8000
http://172.20.10.5:8000
```
## Key Model Implementation Code
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# IMAGE SIZE & CLASS LABELS
img_size = (224, 224)
class_labels = ["Normal", "Precancerous", "Malignant"]

# MODEL 1: EfficientNetB0
base1 = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x1 = GlobalAveragePooling2D()(base1.output)
x1 = Dropout(0.4)(x1)
out1 = Dense(3, activation="softmax")(x1)
eff_model = Model(inputs=base1.input, outputs=out1)

# MODEL 2: Fine-Tuned MobileNetV2
base2 = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x2 = GlobalAveragePooling2D()(base2.output)
x2 = Dropout(0.4)(x2)
out2 = Dense(3, activation="softmax")(x2)
mobile_model = Model(inputs=base2.input, outputs=out2)

# Load trained models
eff_model = tf.keras.models.load_model("efficientnet_model.h5")
mobile_model = tf.keras.models.load_model("mobilenet_finetuned.h5")

# ENSEMBLE LAYER (Soft Voting)
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
eff_output = eff_model(input_layer)
mob_output = mobile_model(input_layer)

ensemble_output = tf.keras.layers.Average()([eff_output, mob_output])
ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)

# Save Final Model
ensemble_model.save("ensemble_model.keras")
print("Final Ensemble Model Saved Successfully")

# IMAGE PREPROCESSING & PREDICTION
def preprocess_image(path):
    img = image.load_img(path, target_size=img_size)
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(img_path):
    img = preprocess_image(img_path)
    eff_pred = eff_model.predict(img)
    mob_pred = mobile_model.predict(img)
    final_prob = (eff_pred + mob_pred) / 2
    pred_class = np.argmax(final_prob)
    print("\nPredicted Class:", class_labels[pred_class])
    print("Confidence Score:", round(float(np.max(final_prob)) * 100, 2), "%")
```

## Results

The performance of the individual models and the ensemble model was
evaluated using accuracy, precision, recall, F1-score, and confusion
matrix.

<img width="450" height="350" alt="model_performance_comparison_small" src="https://github.com/user-attachments/assets/7a89d963-b902-4f35-b0a6-e9b1cc340e27" />

Although MobileNetV2 achieved a higher overall accuracy of 96%, its
performance was not equally balanced across all three classes.
EfficientNetB0 achieved 63% accuracy but showed comparatively higher
misclassification.

The soft-voting ensemble achieved 95% overall accuracy while providing
more balanced classification performance across the Normal,
Precancerous, and Malignant classes. The ensemble achieved an overall
weighted precision, recall, and F1-score of 0.95.

Therefore, the ensemble model was selected for deployment because the
objective was not only to maximize overall accuracy, but also to obtain
consistent classification performance across all three classes (Normal, Precancerous, Malignant).

This system enables early-stage detection, which may support medical professionals and improve preventive healthcare outcomes.

## Output

#### Web-page asking for input from user

<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/e21ff2d8-89cd-4469-b462-9ce465741c55" />

<img width="1920" height="1080" alt="6" src="https://github.com/user-attachments/assets/e7dca178-195a-4242-9e54-f4b260368c87" />

#### Web-page displays the result

<img width="1920" height="1080" alt="7" src="https://github.com/user-attachments/assets/6daef34a-edc8-4aa4-abd0-e2e387a7243a" />

## Future Enhancements

🔹 Store patient history using Firebase/MongoDB

🔹 Add batch image prediction

🔹 Deploy inference on GPU cloud

## References

[1] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural 
networks,” Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

[2] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and 
H. Adam, “MobileNets: Efficient convolutional neural networks for mobile vision applications,” 
arXiv preprint arXiv:1704.04861, 2017. 

[3] J. Zhang, F. Xie, Y. Qian, and X. Xie, “Cervical cancer diagnosis using deep convolutional 
neural networks,” IEEE Access, vol. 8, pp. 91245–91256, 2020. 

[4] K. Zhou and X. Chen, “Ensemble learning for medical image classification: A comprehensive 
review,” Medical Image Analysis, 2021.
