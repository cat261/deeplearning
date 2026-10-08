#EXC 1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import get_file
from PIL import Image
import os

model = ResNet50(weights="imagenet")

urls = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
]

images = []

for url in urls:
    file = get_file(os.path.basename(url), url)
    img = Image.open(file).resize((224,224))
    x = preprocess_input(np.array(img))
    images.append(x)

x = np.array(images)

# Display images
for i in range(len(images)):
    plt.subplot(1,2,i+1)
    plt.imshow((x[i]-x[i].min())/(x[i].max()-x[i].min()))
    plt.axis("off")
plt.show()

# Prediction
pred = model.predict(x)

for i in range(len(pred)):
    print("\nImage",i+1)
    for _,name,prob in decode_predictions(pred[i],top=5)[0]:
        print(name,round(prob*100,2),"%")

# Intermediate representation
layer = tf.keras.Model(model.input,model.get_layer("conv3_block4_out").output)
features = layer.predict(x)

print("Input:",x.shape)
print("Feature:",features.shape)
print("Output:",pred.shape)

plt.imshow(features[0,:,:,0],cmap="gray")
plt.title("Intermediate Representation")
plt.axis("off")
plt.show()



#EXC 2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import get_file
from PIL import Image
import os

# Load model
model = ResNet50(weights="imagenet")

# Load image
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
file = get_file("image.jpg", url)
img = Image.open(file).resize((224,224))

x = preprocess_input(np.array(img))
x = np.expand_dims(x, 0)

# Select 3 layers
layers = ["conv1_conv", "conv2_block3_out", "conv4_block6_out"]

# Create model for feature maps
outputs = [model.get_layer(l).output for l in layers]
feature_model = tf.keras.Model(model.input, outputs)

features = feature_model.predict(x)

# Display feature maps
for layer, feature in zip(layers, features):
    print(layer, feature.shape)

    plt.figure(figsize=(10,4))
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(feature[0,:,:,i], cmap="gray")
        plt.axis("off")
    plt.suptitle(layer)
    plt.show()

# Visualize first-layer filters
filters = model.get_layer("conv1_conv").get_weights()[0]

plt.figure(figsize=(10,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(filters[:,:,0,i], cmap="gray")
    plt.axis("off")
plt.suptitle("Convolution Filters")
plt.show()




#EXC 3

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import os

# 1. Load pretrained CNN without classifier
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Images and labels
urls = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
]

labels = [0, 1, 0, 1]
X = []

# 3. Feature extraction
for url in urls:
    file = get_file(os.path.basename(url), url)
    img = Image.open(file).resize((224,224))
    img = preprocess_input(np.array(img))
    X.append(img)

X = np.array(X)
X = model.predict(X)

print("Feature vector:", X.shape)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, random_state=42
)

# 5. Classification layer
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Prediction
y_pred = clf.predict(X_test)

# 7. Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

# 8. Confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
).plot()
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