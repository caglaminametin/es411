import yaml
import tensorflow as tf
from tensorflow.keras import optimizers
import json

# Yardımcı modüller
from utils.seed_utils import set_seeds
from model.classifier import create_cnn_model
from data.data_generators import get_data_generators
from utils.callbacks import get_callbacks
from utils.plotting import plot_training
from utils.evaluation import evaluate_model

# Config dosyasını yükleme
def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# Ana fonksiyon
if __name__ == "__main__":
    # Config dosyasını yükle
    config = load_config()

    # Seed ayarı
    set_seeds(config["general"]["random_seed"])

    # Eğitim parametreleri
    input_shape = tuple(config["training"]["input_shape"])
    num_classes = config["training"]["num_classes"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    # Veri yolları
    train_dir = config["paths"]["train_dir"]
    val_dir = config["paths"]["val_dir"]
    test_dir = config["paths"]["test_dir"]
    checkpoint_path = config["paths"]["checkpoint_path"]

    # Model oluştur
    model = create_cnn_model(input_shape, num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n📌 Model Özeti:\n")
    model.summary()

    # Veri yükleyiciler
    train_gen, val_gen, test_gen = get_data_generators(
        train_dir, val_dir, test_dir,
        img_size=input_shape[:2],
        batch_size=batch_size
    )

    # Model eğitimi
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=get_callbacks(patience, checkpoint_path),
        verbose=1
    )

    # history kaydet
    with open("history.json", "w") as f:
        json.dump(history.history, f)

    # En iyi modeli yükle
    best_model = tf.keras.models.load_model(checkpoint_path)

    # Eğitim grafikleri (yalnızca Colab içinden çalıştırıldığında görünür)
    plot_training(history)

    # Test değerlendirmesi
    test_acc = evaluate_model(best_model, test_gen)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
