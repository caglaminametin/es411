===========================================
 CNN Image Classifier - README
===========================================

📌 Proje Hakkında (Türkçe)
--------------------------
Bu proje, görüntü sınıflandırma (image classification) amacıyla oluşturulmuş özel bir Convolutional Neural Network (CNN) mimarisi içerir. Model, veri artırma (data augmentation) teknikleriyle eğitilir ve en iyi ağırlıklar 'best_model.keras' dosyasına kaydedilir.

📁 Proje Klasör Yapısı:
- config.yaml              → Eğitim ayarları ve veri yolları
- main.py                  → Ana çalıştırma ve değerlendirme betiği

- /model/
    - backbone.py          → Özellik çıkarıcı CNN modeli
    - classifier.py        → Tam model (backbone + classifier)

- /data/
    - data_generators.py   → Eğitim, doğrulama, test veri üreticileri

- /utils/
    - seed_utils.py        → Random seed sabitleyici
    - callbacks.py         → EarlyStopping, ReduceLR, Checkpoint
    - plotting.py          → Eğitim grafikleri
    - evaluation.py        → Test sonucu, confusion matrix, rapor

🚀 Nasıl Çalıştırılır:

1. config.yaml içindeki veri yollarını güncelle (Google Drive vb.)

2. Ana script’i çalıştır:
   python main.py

Eğitim sonrası loss/accuracy grafiklerini çizer ve en iyi modeli 'best_model.keras' olarak kaydeder.

-------------------------------------------

📌 About the Project (English)
------------------------------
This project implements a custom Convolutional Neural Network (CNN) for image classification. It uses data augmentation and callbacks like EarlyStopping and ModelCheckpoint to improve training.

📁 Project Structure:
- config.yaml              → Training configuration and dataset paths
- main.py                  → Main training and evaluation script

- /model/
    - backbone.py          → Feature extraction CNN
    - classifier.py        → Full model with classification head

- /data/
    - data_generators.py   → ImageDataGenerator setup

- /utils/
    - seed_utils.py        → Random seed fixer
    - callbacks.py         → EarlyStopping, ReduceLR, Checkpoint
    - plotting.py          → Draw training graphs
    - evaluation.py        → Confusion matrix and report generation

🚀 How to Run:

1. Update data paths in config.yaml

2. Run the training:
   python main.py

At the end of training, it will save the best model as 'best_model.keras' and visualize performance.
