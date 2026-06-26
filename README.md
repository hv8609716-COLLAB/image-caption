# image-caption


# Image Caption Generator using InceptionV3

This project uses a pretrained **InceptionV3** deep learning model from TensorFlow to classify an image and generate a predicted caption label.

## Features

* Load image from URL
* Image preprocessing using TensorFlow
* Prediction using pretrained InceptionV3
* Display image with predicted label
* Top predictions with confidence score

## Technologies Used

* Python
* TensorFlow
* NumPy
* Matplotlib
* Pillow (PIL)
* Requests

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_LINK
```

Move into project folder:

```bash
cd YOUR_PROJECT_NAME
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib pillow requests
```

## Run Project

Execute:

```bash
python app.py
```

## Example Output

```text
Predictions:
dog - 96.12%
golden_retriever - 2.80%
Labrador - 0.90%
```

A window will open displaying the image with predicted caption.

## Project Structure

```text
project/
│
├── app.py
├── README.md
└── requirements.txt
```

## Future Improvements

* Upload local images
* Support multiple image models
* Build web interface using Flask
* Add image caption sentence generation

## Author

Harsh Vardhan
