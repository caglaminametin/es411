from tensorflow.keras import callbacks

def get_callbacks(patience=10, checkpoint_path='best_model.keras'):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    return [early_stop, reduce_lr, checkpoint]
