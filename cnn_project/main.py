import yaml
from utils.seed_utils import set_seeds
from model.classifier import create_cnn_model
from data.data_generators import get_data_generators
from utils.callbacks import get_callbacks
from utils.plotting import plot_training
from utils.evaluation import evaluate_model

import tensorflow as tf
from tensorflow.keras import optimizers

def load_config(path='config.yaml'):
    with open(path) as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config()
    set_seeds(config["general"]["random_seed"])

    input_shape = tuple(config["training"]["input_shape"])
    num_classes = config["training"]["num_classes"]

    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    train_gen, val_gen, test_gen = get_data_generators(
        config["paths"]["train_dir"],
        config["paths"]["val_dir"],
        config["paths"]["test_dir"],
        img_size=input_shape[:2],
        batch_size=config["training"]["batch_size"]
    )

    history = model.fit(
        train_gen,
        epochs=config["training"]["epochs"],
        validation_data=val_gen,
        callbacks=get_callbacks(config["training"]["patience"], config["paths"]["checkpoint_path"])
    )

    # En iyi modeli yükle
    best_model = tf.keras.models.load_model(config["paths"]["checkpoint_path"])

    plot_training(history)
    acc = evaluate_model(best_model, test_gen)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
