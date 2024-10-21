# Skin Cancer Detection using CNN

> A deep learning project to accurately detect melanoma using a custom Convolutional Neural Network (CNN). This model aims to assist dermatologists by evaluating skin images and identifying the presence of melanoma, potentially reducing manual effort in diagnosis.

## Table of Contents
* [General Information](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- **Background:** Melanoma is a severe form of skin cancer responsible for 75% of skin cancer deaths. Early detection is crucial for effective treatment. Automating the detection process using deep learning can assist medical professionals and potentially save lives.

- **Business Problem:** The manual diagnosis of melanoma from skin images is time-consuming and requires expert knowledge. By developing an accurate CNN model for melanoma detection, we can provide a tool that alerts dermatologists to potential cases, streamlining the diagnostic process and improving patient outcomes.

- **Dataset Used:** The dataset consists of 2,357 images of malignant and benign skin lesions, sourced from the International Skin Imaging Collaboration (ISIC). It includes nine classes of skin diseases:

  1. **Actinic keratosis**
  2. **Basal cell carcinoma**
  3. **Dermatofibroma**
  4. **Melanoma**
  5. **Nevus**
  6. **Pigmented benign keratosis**
  7. **Seborrheic keratosis**
  8. **Squamous cell carcinoma**
  9. **Vascular lesion**

- **Project Objective:** Build a CNN model that can accurately classify images into one of the nine skin cancer types, with a particular focus on detecting melanoma.

## Conclusions
- **Initial Model Findings:**
  - The first CNN model trained for 20 epochs showed signs of overfitting.
  - Training accuracy was significantly higher than validation accuracy.
  - The model did not generalize well to unseen data.

- **Data Augmentation:**
  - Applied data augmentation techniques (random flip, rotation, zoom) to increase data diversity.
  - Overfitting was reduced, and validation accuracy improved.
  - The model began to generalize better, but there was still room for improvement.

- **Class Imbalance Handling:**
  - Analysis revealed class imbalances in the dataset.
  - Used the Augmentor library to balance classes by augmenting underrepresented classes to have at least 500 images each.
  - Retrained the model on the balanced dataset for 30 epochs.
  - Validation accuracy improved to approximately **74%**, and overfitting was further reduced.

- **Final Model Performance:**
  - Achieved a training accuracy of **94.06%** and a validation accuracy of **73.96%**.
  - The confusion matrix and classification report indicated improved performance across all classes.
  - The model demonstrated better generalization and reduced bias towards overrepresented classes.

- **Recommendations:**
  - Implement early stopping and regularization techniques to prevent potential overfitting in later epochs.
  - Further hyperparameter tuning (e.g., learning rate adjustments) could enhance model performance.
  - Explore the use of more complex architectures or pre-trained models for potential improvement.

## Technologies Used
- **Python 3.8**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Augmentor**
- **scikit-learn**

## Acknowledgements
- **Dataset:** Thanks to the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/) for providing the skin cancer image dataset.
- **Inspiration:** This project was inspired by the critical need for early detection of melanoma to reduce mortality rates associated with skin cancer.
- **References:**
  - TensorFlow Tutorials: [Image Classification](https://www.tensorflow.org/tutorials/images/classification)
  - Augmentor Library Documentation: [Augmentor](https://augmentor.readthedocs.io/en/master/)
  - Keras Documentation: [Keras API Reference](https://keras.io/api/)
  - Scikit-learn Documentation: [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Contact
Created by [sahil-awasthi](https://github.com/sahil-awasthi) - feel free to contact me!

