# TEAM.NO.49:AI Based Multi-Class Ensemble Learning System for Cervical Cancer Detection

## About

This project focuses on detecting cervical cancer using a deep learning ensemble architecture. The system classifies cervical cytology images into:

1.Normal

2.Precancerous

3.Malignant


The solution integrates EfficientNetB0 and MobileNetV2 (fine-tuned), combined using soft probability voting, achieving a final accuracy of 95%.
A Flask-based web app enables users to upload images and receive predictions, along with a downloadable medical-style PDF report.

## Features

📤 Upload cervical cell image

🤖 Real-time prediction using ensemble deep learning

📊 Confidence score for each class

📝 Enter patient details before prediction

📄 Downloadable PDF medical report

## Development Requirements

<img width="692" height="317" alt="1" src="https://github.com/user-attachments/assets/7270de6f-05f0-4595-84d5-f73619c425e1" />

## System Architecture

![2](https://github.com/user-attachments/assets/2759b7c0-b629-4076-a664-a58951980712)


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

The ensemble model demonstrated improved performance across all classes compared to individual models.

The final deployed model achieved:

<img width="507" height="346" alt="3" src="https://github.com/user-attachments/assets/239ed591-3dfb-46c6-97df-125076b88d1b" />

<img width="676" height="215" alt="4" src="https://github.com/user-attachments/assets/d1c1468c-8306-480f-be26-2dccdf3be891" />

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
The final ensemble model achieved an accuracy of 95%, providing strong performance in classifying cervical cytology images across all three classes (Normal, Precancerous, Malignant).

This system enables early-stage detection, which may support medical professionals and improve preventive healthcare outcomes.

## Output

#### Web-page asking for input from user

<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/519c4f74-6dba-42c4-9d48-ee570f977623" />

<img width="1920" height="1080" alt="6" src="https://github.com/user-attachments/assets/d106170a-9322-4847-9dba-37a71180f197" />

#### Web-page displays the result

<img width="1920" height="1080" alt="7" src="https://github.com/user-attachments/assets/b92e634b-75d8-4d4a-9eb4-6d555ccdea10" />

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

