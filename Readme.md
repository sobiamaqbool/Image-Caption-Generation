# Image Captioning with CNN-RNN (Flickr8k Dataset)

This project implements an image captioning model using a Convolutional Neural Network (CNN) as the encoder and a Recurrent Neural Network (RNN) as the decoder. The model is trained on the Flickr8k dataset to generate textual descriptions for images.

## Model Architecture

Encoder: Pretrained ResNet-18 (excluding the classification layer), followed by a linear transformation to the embedding space.
Decoder: LSTM with an embedding layer and a linear output layer that predicts the next word in the sequence.
Vocabulary: Custom vocabulary builder with a frequency threshold for including words.

## Requirements

Python 3.7+
PyTorch
Torchvision
NLTK
Pillow (PIL)
tqdm
NumPy
Install dependencies using:
pip install -r requirements.txt

## Training

Download the NLTK tokenizer
Build the vocabulary from all captions
Load and preprocess image-caption pairs
Train the encoder and decoder for 20 epochs
Save the trained model to caption.pth

## Model Output

During training, the script prints the average loss per epoch. After training, the model checkpoint contains:
Encoder weights
Decoder weights
Vocabulary (word2idx mapping)
This can be used for inference or further training.

## License

This project is intended for academic and educational purposes. Make sure you adhere to the licensing terms of the Flickr8k dataset.


