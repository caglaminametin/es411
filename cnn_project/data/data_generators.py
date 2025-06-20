from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, test_dir, img_size=(128,128), batch_size=32):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb'
    )
    val = val_test_gen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb'
    )
    test = val_test_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    return train, val, test
