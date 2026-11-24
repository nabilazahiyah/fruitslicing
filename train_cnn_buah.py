import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SIZE = 180
NUM_CLASSES = 6  # Semangka, Jeruk, Buah Naga, Lemon, Apel Merah, Kiwi

# 1. Augmentasi & Preprocessing Dataset 6 Kelas Buah
# Pastikan folder dataset_buah/train berisi 6 subfolder sesuai nama kelas buah
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'dataset_buah/train',
    target_size=(SIZE, SIZE),
    batch_size=32,
    class_mode='categorical',
    seed=79
)

# 2. Arsitektur CNN Multi-Kelas
def cnn_fruit_classifier(size=180, num_classes=6):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(size, size, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        # Layer output 6 kelas dengan aktivasi Softmax
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    model = cnn_fruit_classifier(SIZE, NUM_CLASSES)
    model.summary()