# %%
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from pathlib import Path

# %%
original_dataset_dir = Path('Data/dogs-vs-cats/PetImages')

# %%
base_dir = Path('Data/dogs-vs-cats/PetImages/cats_and_dogs_small')
try:
    os.mkdir(base_dir)
except FileExistsError:
    pass

# %%
# Directories for the training, validation, and test splits
train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

sup_list = [train_dir, validation_dir, test_dir]

for dir in sup_list:
    if not os.path.exists(dir):
        os.mkdir(dir)

# %%
train_cats_dir = train_dir / 'cats'
train_dogs_dir = train_dir / 'dogs'
validation_cats_dir = validation_dir / 'cats'
validation_dogs_dir = validation_dir / 'dogs'
test_cats_dir = test_dir / 'cats'
test_dogs_dir = test_dir / 'dogs'

sup_list = [
    train_cats_dir,
    train_dogs_dir,
    validation_cats_dir,
    validation_dogs_dir,
    test_cats_dir,
    test_dogs_dir,
]

for dir in sup_list:
    if not os.path.exists(dir):
        os.mkdir(dir)

# %%
fnames = [f'{i}.jpg' for i in range(4000)]
for fname in fnames:
    src = original_dataset_dir / 'Cat' / fname
    dst = train_cats_dir / fname
    shutil.copyfile(src, dst)

fnames = [f'{i}.jpg' for i in range(4000, 5000)]
for fname in fnames:
    src = original_dataset_dir / 'Cat' / fname
    dst = validation_cats_dir / fname
    shutil.copyfile(src, dst)

fnames = [f'{i}.jpg' for i in range(5000, 6000)]
for fname in fnames:
    src = original_dataset_dir / 'Cat' / fname
    dst = test_cats_dir / fname
    shutil.copyfile(src, dst)

# %%
fnames = [f'{i}.jpg' for i in range(4000)]
for fname in fnames:
    src = original_dataset_dir / 'Dog' / fname
    dst = train_dogs_dir / fname
    shutil.copyfile(src, dst)

fnames = [f'{i}.jpg' for i in range(4000, 5000)]
for fname in fnames:
    src = original_dataset_dir / 'Dog' / fname
    dst = validation_dogs_dir / fname
    shutil.copyfile(src, dst)

fnames = [f'{i}.jpg' for i in range(5000, 6000)]
for fname in fnames:
    src = original_dataset_dir / 'Dog' / fname
    dst = test_dogs_dir / fname
    shutil.copyfile(src, dst)

# %%
conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))

# %%
base_dir = Path('Data/dogs-vs-cats/PetImages/cats_and_dogs_small')
train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

# %%
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# %%
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# %%
train_features, train_labels = extract_features(train_dir, 8000)
validation_features, validation_labels = extract_features(validation_dir, 2000)
test_features, test_labels = extract_features(test_dir, 2000)

# %%
train_features.shape

# %%
train_features = np.reshape(train_features, (8000, 4*4*512))
validation_features = np.reshape(validation_features, (2000, 4*4*512))
test_features = np.reshape(test_features, (2000, 4*4*512))

# %%
from tensorflow.keras import models
from tensorflow.keras import layers

# %%
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# %%
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

# %%
history = model.fit(train_features, train_labels,
                   epochs=15,
                   batch_size=20,
                   validation_data = (validation_features, validation_labels))

# %%
import plotly.graph_objects as go

# %%
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = list(range(1, len(acc)+1))

# %%
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=epochs,
    y=acc,
    mode='lines',
    name='Accuracy trace'
))

fig.add_trace(go.Scatter(
    x=epochs,
    y=val_acc,
    mode='lines',
    name='Validation accuracy trace'
))

fig.update_layout(
    template='plotly_dark'
)

fig.show()

# %%
results = model.evaluate(test_features, test_labels, batch_size=100)

# %%
model.save('cats_and_dogs_model.h5')

# %%



