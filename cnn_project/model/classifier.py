from tensorflow.keras import layers, models
from .backbone import build_backbone

def create_cnn_model(input_shape, num_classes):
    backbone = build_backbone(input_shape)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=backbone.input, outputs=outputs)
    return model
