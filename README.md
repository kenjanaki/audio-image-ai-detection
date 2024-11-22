# audio-image-ai-detection
ML/DL project to detect probability of AI in audio and images

## Audio Deepfake Detection

### Problem Statement:
Develop a reliable system to detect synthetic (AI-generated) audio, which can often be misused for fraudulent purposes such as identity theft, fake news, and phishing scams.

### Challenges:
- Mimicking real voices with high fidelity through advanced TTS models.
- Lack of temporal and spectral consistency in some synthetic audio.
- Balancing model efficiency and computational resource requirements.

### Available Techniques:
- Feature-Based Methods: Extraction of features like MFCCs, chroma, and ZCR for classification.
- Deep Learning Models: Utilizing XGBoost, LightGBM, or neural networks for feature analysis.
- Spectral Analysis: Mel-spectrograms and CQT spectrograms for visual representation of audio data.

### Proposed Methodology:
- Preprocess audio by normalizing and truncating samples to a uniform duration.
- Extract key features such as MFCCs, chroma, ZCR, and mel spectrograms using libraries like Librosa.
- Train an XGBoost classifier on these features to distinguish between real and synthetic audio.

### Tech Stack:
- Software platform: Python
- Libraries: Librosa, NumPy, SciPy, XGBoost


## AI-Generated Image Detection

### Problem Statement:
Design a model to differentiate between AI-generated and human-created images, addressing the challenges of identifying artifacts in synthetic images.

### Challenges:
- Identifying subtle inconsistencies or artifacts introduced by generative models like StyleGAN.
- Scalability when handling large datasets of diverse images.
- Need for models that can generalize across various AI generation techniques.

### Available Techniques:
- Histogram-Based Methods: Analyzing pixel intensity distributions.
- Convolutional Neural Networks (CNNs): Detecting patterns and artifacts in image data.
- Transformer-Based Models: Leveraging fine-tuned transformers for image classification tasks.

### Proposed Methodology:
- Preprocess images with rescaling and augmentation techniques.
- Use a CNN-based architecture (e.g., VGG16) fine-tuned on labeled datasets of real and fake images.
- Evaluate the model using metrics like accuracy, precision, and ROC-AUC.

### Tech Stack:
- Software platform: Python
- Libraries: TensorFlow, Keras, OpenCV, NumPy
