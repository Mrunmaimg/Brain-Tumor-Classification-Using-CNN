# Brain Tumor Classification Using CNN
 

This project is designed to classify brain tumors using Convolutional Neural Networks (CNN) based on MRI images. The application provides a web interface where users can upload MRI images and get predictions about whether a brain tumor is present or not.

## Features
- **Model**: A CNN model trained from scratch (not pre-trained) specifically for this task.(BrainTumor10EpochsCategorical.h5)
- **Web Interface**: Built using Flask to allow users to upload MRI images for prediction.
- **Real-time Predictions**: Classifies MRI images into two categories: "No Brain Tumor" or "Yes Brain Tumor".
- **Training**: CNN model trained using a dataset of MRI images categorized into two classes: with and without brain tumors.

## File Structure
- `app.py`: The main Flask web application for uploading MRI images and getting predictions.
- `maintest.py`: Script to load and test the model with a sample MRI image.
- `train.py`: Code for training the CNN model using a dataset of brain tumor images.
- `BrainTumor10EpochsCategorical.h5`: The CNN model file created after running `train.py`, which is used for predictions in the web app.

## Requirements
- TensorFlow
- Keras
- Flask
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- Scikit-learn
- Werkzeug

## Dataset

The dataset used for training the brain tumor detection model is publicly available on Kaggle. It contains MRI images classified into two categories: "No Brain Tumor" and "Yes Brain Tumor".

You can find the dataset here: [Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Mrunmaimg/Brain-Tumor-Classification-Using-CNN.git
    ```


2. Run the Flask application:
    ```bash
    python app.py
    ```

3. Access the app in your browser at: `http://127.0.0.1:5000/`.

## Usage
- Navigate to the web interface, upload an MRI image, and click "Predict" to get the classification result.
- The result will be either "No Brain Tumor" or "Yes Brain Tumor" based on the input image.

## Model Training (train.py)
- The `train.py` file handles the model training. It uses a dataset of brain tumor images split into "yes" (with tumor) and "no" (without tumor) categories.
- The images are resized to 64x64 pixels and normalized.
- The CNN model includes several convolutional layers, max-pooling, and dropout layers for efficient training and to prevent overfitting.
- The model is compiled using categorical cross-entropy loss and the Adam optimizer, trained for 10 epochs.
- After running `train.py`, the trained model is saved as `BrainTumor10EpochsCategorical.h5`.

## Sample Test (maintest.py)
- `maintest.py` demonstrates how to use the trained model to predict the presence of a brain tumor from a single image.

## Dataset
The dataset used for training is stored in the `datasets/` folder, containing two subfolders:
- `yes/`: MRI images of brain tumors.
- `no/`: MRI images without brain tumors.

## Example Prediction
1. Upload an MRI image using the web interface.
2. The application will preprocess the image and classify it using the loaded CNN model.
3. The result will be displayed as "No Brain Tumor" or "Yes Brain Tumor".
   ![result_yes](https://github.com/Mrunmaimg/Brain-Tumor-Classification-Using-CNN/blob/main/result_yes.jpeg)
   ![result_no](https://github.com/Mrunmaimg/Brain-Tumor-Classification-Using-CNN/blob/main/result_no.jpeg)


## Model Performance

The model was trained for 10 epochs, achieving high accuracy and low loss during both training and validation. Below is the accuracy and loss per epoch:

![Training and Validation Accuracy](https://github.com/Mrunmaimg/Brain-Tumor-Classification-Using-CNN/blob/main/accuracy.png)

- **Final Training Accuracy**: 99.34%
- **Final Validation Accuracy**: 97.67%
- **Final Training Loss**: 0.0295
- **Final Validation Loss**: 0.0763

The model performed exceptionally well, with over 97% validation accuracy by the end of the training process.



## Contribute

Feel free to fork this repository, explore the code, and contribute to its improvement! Whether it’s optimizing the model, improving accuracy, or adding new features, all contributions are welcome.

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

Let's collaborate and make this project even better together!





## Conclusion
This project provides a simple and efficient solution for brain tumor classification using CNN. The model is trained from scratch, which allows for flexibility in improving its performance with more data or different training techniques.

