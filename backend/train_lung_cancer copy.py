import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Define Constants
IMG_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 16
DATA_DIR = r"C:\Users\PRASANNA SAXENA\Downloads\archive\LIDC-IDRI-slices"
MODEL_PATH = 'lung_cancer_cnn_model.keras'

def load_png_image(img_path, img_size=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess a PNG image."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values (0-1)
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def get_label_from_path(img_path):
    """Extract label from image path or filename more reliably."""
    # This is a placeholder - you need to adapt this to your specific dataset structure
    # For LIDC-IDRI, you might need to parse metadata files or use a mapping file
    # Here's a simple example based on folder structure
    if 'nodule' in img_path.lower() or 'malignant' in img_path.lower():
        return 1
    elif 'non-nodule' in img_path.lower() or 'benign' in img_path.lower():
        return 0
    else:
        # If you can't determine from path, you might need to use a mapping file
        # For demonstration, we'll use a random assignment (you should NOT do this in practice)
        # return np.random.choice([0, 1])
        # Instead, skip images with uncertain labels
        return None

def load_data(data_dir):
    """Load PNG images into arrays with proper error handling."""
    images, labels, paths = [], [], []
    
    # Count total files for progress tracking
    total_files = sum([len(files) for _, _, files in os.walk(data_dir) if any(f.endswith('.png') for f in files)])
    processed = 0
    
    print(f"Starting to load {total_files} potential image files...")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                label = get_label_from_path(img_path)
                
                # Skip images with uncertain labels
                if label is None:
                    continue
                
                img = load_png_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    paths.append(img_path)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{total_files} images...")
    
    if not images:
        raise ValueError("No valid PNG images found! Check dataset path and structure.")
    
    print(f"Successfully loaded {len(images)} images.")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return np.array(images), np.array(labels), paths

def create_model():
    """Create a CNN model with appropriate regularization."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile with appropriate metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

def plot_training_history(history):
    """Plot the training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with appropriate metrics."""
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return y_pred, y_pred_prob

def main():
    print("Loading data...")
    X, y, paths = load_data(DATA_DIR)
    
    # Handle class imbalance if needed
    class_counts = np.bincount(y)
    if min(class_counts) / max(class_counts) < 0.25:
        print(f"Warning: Significant class imbalance detected. Class counts: {class_counts}")
        # Consider implementing class weights or sampling strategies
    
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    # Create stratified train/val/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Train set: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
    print(f"Validation set: {X_val.shape}, Class distribution: {np.bincount(y_val)}")
    print(f"Test set: {X_test.shape}, Class distribution: {np.bincount(y_test)}")
    
    # Create data augmentation generator
    data_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Create and train model
    model = create_model()
    print(model.summary())
    
    print("\nTraining model...")
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Save predictions for further analysis
    results_df = pd.DataFrame({
        'path': paths[-len(y_test):],
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability': y_pred_prob.flatten()
    })
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()



