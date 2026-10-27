#EXC 1
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions
)
from tensorflow.keras.models import Model

DATASET_PATH = Path(r"D:\ImageNet")
NUM_CLASSES = 10
IMAGES_PER_CLASS = 100
random.seed(42)

# Select only 10 folders; scan images only inside those folders
folders = sorted([p for p in DATASET_PATH.iterdir() if p.is_dir()])
selected_folders = folders[:NUM_CLASSES]

classes = []
for folder in selected_folders:
    imgs = [p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    random.shuffle(imgs)
    imgs = imgs[:IMAGES_PER_CLASS]
    if len(imgs) >= 2:
        classes.append((folder.name, imgs))

train_files, test_files = [], []
for label, imgs in classes:
    tr, te = train_test_split(imgs, test_size=0.2, random_state=42)
    train_files.extend(tr)
    test_files.extend(te)

print("Selected classes:", [c[0] for c in classes])
print("Training images:", len(train_files))
print("Testing images:", len(test_files))

# Load pretrained VGG16
model = VGG16(weights="imagenet")
feature_model = Model(
    model.input, model.get_layer("block3_pool").output
)

# Test 5 images
for path in random.sample(test_files, min(5, len(test_files))):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img)
    batch = preprocess_input(np.expand_dims(arr, axis=0))

    pred = model.predict(batch, verbose=0)
    top5 = decode_predictions(pred, top=5)[0]
    features = feature_model.predict(batch, verbose=0)

    print("\nActual folder label:", path.parent.name)
    print("Input shape:", batch.shape)
    print("Feature-map shape:", features.shape)
    print("Top-5 predictions:")
    for _, label, score in top5:
        print(f"{label}: {score*100:.2f}%")

    print("Top-1 prediction:", top5[0][1])

    plt.imshow(img)
    plt.title(path.parent.name)
    plt.axis("off")
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(features[0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.suptitle("VGG16 Feature Maps")
    plt.tight_layout()
    plt.show()

#EXC 2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# 1. Load an ImageNet image
img_path = r"D:\ImageNet\golden_retriever\dog.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
batch = preprocess_input(np.expand_dims(x, axis=0))

# 2. Load pretrained VGG16
model = VGG16(weights="imagenet")

# 3. Select early, intermediate and deep convolutional layers
layer_names = ["block1_conv1", "block3_conv1", "block5_conv1"]
outputs = [model.get_layer(name).output for name in layer_names]
feature_model = Model(inputs=model.input, outputs=outputs)

# 4. Extract feature maps
features = feature_model.predict(batch, verbose=0)

# 5. Visualize feature maps and print dimensions
for name, fmap in zip(layer_names, features):
    print(name, "feature-map dimensions:", fmap.shape)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fmap[0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.suptitle(name + " - Feature Maps")
    plt.tight_layout()
    plt.show()

# 6. Visualize learned convolutional filters
for name in ["block1_conv1", "block3_conv1", "block5_conv1"]:
    weights, biases = model.get_layer(name).get_weights()
    print(name, "filter dimensions:", weights.shape)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        filt = weights[:, :, :, i]
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        ax.imshow(filt)
        ax.axis("off")
    plt.suptitle(name + " - Learned Filters")
    plt.tight_layout()
    plt.show()

# 7. Display original input image
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()


#EXC 3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# 1. Dataset: folders contain class names and images
DATASET_PATH = Path(r"D:\ImageNet")
NUM_CLASSES = 10
IMAGES_PER_CLASS = 100
np.random.seed(42)

# 2. Select 10 classes and limit images
folders = sorted([p for p in DATASET_PATH.iterdir() if p.is_dir()])[:NUM_CLASSES]
X, labels = [], []

for folder in folders:
    imgs = [p for p in folder.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    np.random.shuffle(imgs)

    for path in imgs[:IMAGES_PER_CLASS]:
        img = image.load_img(path, target_size=(224, 224))
        X.append(image.img_to_array(img))
        labels.append(folder.name)

X = np.array(X, dtype=np.float32)
labels = np.array(labels)

# 3. Encode class labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y_cat = to_categorical(y, num_classes=len(encoder.classes_))

print("Classes:", encoder.classes_)
print("Images:", len(X))

# 4. Load pretrained ResNet50 without original classifier
base = ResNet50(weights="imagenet", include_top=False,
                input_shape=(224, 224, 3), pooling="avg")

# 5. Extract high-level features in batches
features = base.predict(preprocess_input(X.copy()), batch_size=16, verbose=1)
print("Feature-vector dimensions:", features.shape)

# 6. Store extracted features
np.save("features.npy", features)
np.save("labels.npy", y)

# 7. Split into training and testing sets
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test = features[train_idx], features[test_idx]
y_train, y_test = y_cat[train_idx], y[test_idx]

# 8. Train a new classification layer
classifier = __import__("tensorflow").keras.Sequential([
    Dense(128, activation="relu", input_shape=(features.shape[1],)),
    Dense(len(encoder.classes_), activation="softmax")
])

classifier.compile(optimizer="adam",
                   loss="categorical_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# 9. Test on unseen images
pred = np.argmax(classifier.predict(X_test, verbose=0), axis=1)

# 10. Calculate evaluation metrics
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_test, pred, average="macro", zero_division=0))
print("F1-score:", f1_score(y_test, pred, average="macro", zero_division=0))

# 11. Confusion matrix
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(
    xticks_rotation=90, cmap="Blues"
)
plt.title("ResNet50 Feature Classifier")
plt.tight_layout()
plt.show()




#EXC 4

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import get_file
from PIL import Image

# 1. CNN encoder
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Images and captions
urls = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
]

captions = [
    "a photo of a sunflower",
    "a photo of a person"
]

# 3. Extract CNN features
X = []

for url in urls:
    file = get_file(url.split("/")[-1], url)
    img = Image.open(file).resize((224,224))
    img = preprocess_input(np.array(img))
    X.append(img)

X = cnn.predict(np.array(X))

# 4. Add START and END tokens
captions = ["<start> " + c + " <end>" for c in captions]

# 5. Tokenize
tok = Tokenizer()
tok.fit_on_texts(captions)

seq = tok.texts_to_sequences(captions)
seq = pad_sequences(seq, padding="post")

vocab = len(tok.word_index) + 1

# 6. Input and output sequences
decoder_in = seq[:, :-1]
decoder_out = seq[:, 1:]

# 7. LSTM decoder
image = Input(shape=(2048,))
text = Input(shape=(None,))

x = Dense(128, activation="relu")(image)
x = tf.keras.layers.RepeatVector(decoder_in.shape[1])(x)
x = LSTM(128, return_sequences=True)(x)
output = Dense(vocab, activation="softmax")(x)

model = Model([image, text], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 8. Train
model.fit(
    [X, decoder_in],
    np.expand_dims(decoder_out, -1),
    epochs=10
)

print("Model trained")
print("Vocabulary:", tok.word_index)






#EXC 5

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from tensorflow.keras.models import Model

# 1. Text descriptions
texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a tiger"
]

# 2. Simple text representation
words = ["dog", "cat", "tiger"]
text_vectors = np.eye(3)

# 3. Generator
noise = Input(shape=(100,))
text = Input(shape=(3,))

x = Concatenate()([noise, text])
x = Dense(128, activation="relu")(x)
x = Dense(28*28, activation="sigmoid")(x)
output = Reshape((28,28,1))(x)

G = Model([noise, text], output)

# 4. Discriminator
image = Input(shape=(28,28,1))
text = Input(shape=(3,))

x = Flatten()(image)
x = Concatenate()([x, text])
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

D = Model([image, text], output)

D.compile(optimizer="adam", loss="binary_crossentropy")

# 5. Combined GAN
D.trainable = False
noise = Input(shape=(100,))
text = Input(shape=(3,))

fake = G([noise, text])
valid = D([fake, text])

GAN = Model([noise, text], valid)
GAN.compile(optimizer="adam", loss="binary_crossentropy")

# 6. Train
for epoch in range(100):
    noise = np.random.randn(3,100)
    fake = G.predict([noise, text_vectors], verbose=0)

    real = np.random.rand(3,28,28,1)

    D.train_on_batch([real,text_vectors], np.ones((3,1)))
    D.train_on_batch([fake,text_vectors], np.zeros((3,1)))

    GAN.train_on_batch(
        [np.random.randn(3,100), text_vectors],
        np.ones((3,1))
    )

# 7. Generate image
noise = np.random.randn(1,100)
text = np.array([[1,0,0]])       # dog

img = G.predict([noise,text], verbose=0)

plt.imshow(img[0,:,:,0], cmap="gray")
plt.title("Generated Image: Dog")
plt.axis("off")
plt.show()


#4 new
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load images from 5 class folders
DATASET = Path(r"D:\ImageNet")
images, captions = [], []

for folder in sorted(DATASET.iterdir())[:5]:
    if folder.is_dir():
        for p in list(folder.glob("*.jpg"))[:10]:
            images.append(p)
            captions.append(
                "<start> a photo of " + folder.name.replace("_", " ") + " <end>"
            )

# 2. CNN encoder: extract image features
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
X = []
for p in images:
    img = image.load_img(p, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, 0))
    X.append(cnn.predict(arr, verbose=0)[0])
X = np.array(X)

# 3. Tokenize captions and create vocabulary
tok = Tokenizer(filters="", lower=True)
tok.fit_on_texts(captions)
seq = tok.texts_to_sequences(captions)
seq = pad_sequences(seq, padding="post")
vocab = len(tok.word_index) + 1

# 4. Prepare next-word training samples
img_features, inputs, targets = [], [], []
for i, s in enumerate(seq):
    for j in range(1, len(s)):
        if s[j] == 0:
            continue
        img_features.append(X[i])
        inputs.append(s[:j])
        targets.append(s[j])

inputs = pad_sequences(inputs, maxlen=seq.shape[1]-1, padding="post")
targets = tf.keras.utils.to_categorical(targets, num_classes=vocab)
img_features = np.array(img_features)

# 5. CNN-LSTM caption model
image_input = Input(shape=(2048,))
word_input = Input(shape=(inputs.shape[1],))
a = Dense(128, activation="relu")(image_input)
a = tf.keras.layers.RepeatVector(inputs.shape[1])(a)
b = Embedding(vocab, 128, mask_zero=True)(word_input)
x = Add()([a, b])
x = LSTM(128)(x)
out = Dense(vocab, activation="softmax")(x)

model = Model([image_input, word_input], out)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 6. Train the LSTM decoder
model.fit([img_features, inputs], targets, epochs=10, batch_size=16)

# 7. Generate a caption for an image
def caption_image(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    feat = cnn.predict(arr, verbose=0)

    sentence = "<start>"
    for _ in range(12):
        s = tok.texts_to_sequences([sentence])[0]
        s = pad_sequences([s], maxlen=inputs.shape[1], padding="post")
        pred = model.predict([feat, s], verbose=0)[0]
        word = tok.index_word.get(np.argmax(pred), "")
        if not word or word == "<end>":
            break
        sentence += " " + word
    return sentence.replace("<start>", "").strip()

print("Generated caption:", caption_image(images[0]))
