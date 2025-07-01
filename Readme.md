#  Image Caption Generation

A neural image captioning system that generates natural language descriptions of images using encoder-decoder architectures with attention mechanisms.

---

##  Project Overview

This repository implements an image captioning model based on:

- **Encoder**: Pre-trained CNN (e.g., VGG16, InceptionV3, or ResNet) to extract visual features.
- **Decoder**: LSTM (with optional attention) that generates captions word-by-word.
- **Training Data**: Flickr8k dataset (or similar), where each image has multiple associated captions.
- **Attention Mechanism**: Improves performance by focusing on relevant image regions during word generation.

---

##  Features

- Extract image features via pre-trained CNN
- Text preprocessing and tokenization
- Attention-enabled caption generation
- Support for greedy and beam search decoding
- Evaluation with BLEU scores and other metrics
- (Optional) GUI or notebook interface for demonstration

---

##  Tech Stack & Dependencies

- **Python** 3.7+
- **Deep Learning**: TensorFlow/Keras or PyTorch
- **Data Handling**: `numpy`, `pandas`, `pickle`
- **Image Tools**: `Pillow`, `opencv-python`
- **Evaluation**: `nltk` (BLEU), `scikit-learn`
- *(Optional GUI)*: `tkinter` or `streamlit`
- See `requirements.txt` for exact versions

---

##  Setup Guide
1. Clone the repo:
```bash
git clone https://github.com/sobiamaqbool/Image-Caption-Generation.git
cd Image-Caption-Generation

2. Install dependencies:
pip install -r requirements.txt

3. Prepare the dataset:
Download Flickr8k (images + captions.txt)
Organize as:
Flickr8k/
  Images/
  captions.txt
4. Preprocess captions & build tokenizer:
python preprocess_captions.py \
  --captions captions.txt \
  --output_dir data/
5. Extract image features:
python extract_features.py \
  --image_dir Flickr8k/Images \
  --model vgg16 \
  --output features.pkl
6. Train the captioning model:
python train.py \
  --features data/features.pkl \
  --captions data/captions.pkl \
  --epochs 20 \
  --model_out models/cap_model.h5
7. Generate captions:
From images in a folder:
python generate.py \
  --model models/cap_model.h5 \
  --image_dir test_images/
## Evaluation
After training, run:
python evaluate.py \
  --model models/cap_model.h5 \
  --test_features data/test_features.pkl \
  --test_captions data/test_captions.pkl
Outputs BLEU-1 to BLEU-4 scores and sample caption comparisons.
