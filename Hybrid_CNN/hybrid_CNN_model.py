import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing_extract_features import (
    preprocess_image,
    segment_banana,
    color_features,
    texture_features,
    ripeness_factor
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D, 
    Dropout, 
    Input, 
    Concatenate,
    BatchNormalization,
    Multiply
)
from tensorflow.keras.applications import MobileNetV2

class HybridBananaClassifier:
    def __init__(self):
        self.img_size = (224, 224)
        self.num_classes = 4
        self.traditional_features_size = 17
        
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        self.model = self._build_model()
        
    def _build_model(self):
        img_input = Input(shape=(*self.img_size, 3))
        
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=img_input
        )
        
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        x = base_model.output
        spatial_features = GlobalAveragePooling2D()(x)
        
        spatial_features = BatchNormalization()(spatial_features)
        
        traditional_input = Input(shape=(self.traditional_features_size,))
        
        trad_features = Dense(64, activation='relu')(traditional_input)
        trad_features = BatchNormalization()(trad_features)
        trad_features = Dropout(0.3)(trad_features)
        trad_features = Dense(128, activation='relu')(trad_features)
        trad_features = BatchNormalization()(trad_features)
        
        combined = Concatenate()([spatial_features, trad_features])
   
        combined_projected = Dense(256, activation='relu')(combined)
        combined_projected = BatchNormalization()(combined_projected)
        
        attention1 = Dense(256, activation='tanh')(combined_projected)
        attention1 = Dense(256, activation='sigmoid')(attention1)
        attention2 = Dense(256, activation='tanh')(combined_projected)
        attention2 = Dense(256, activation='sigmoid')(attention2)
        
        attention = tf.keras.layers.Average()([attention1, attention2])
        
        attended_features = Multiply()([combined_projected, attention])
       
        x = Dense(256, activation='relu')(attended_features)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=[img_input, traditional_input], outputs=predictions)
        return model
        
    def extract_traditional_features(self, image):
        from preprocessing_extract_features import (
            preprocess_image,
            segment_banana,
            color_features,
            texture_features,
            ripeness_factor
        )
     
        contrast_img, hsv_image, mask = preprocess_image(image)
        
        color_feats = color_features(hsv_image, mask)
        texture_feats = texture_features(contrast_img, mask)
        ripeness = ripeness_factor(hsv_image, mask)

        features = color_feats + [
            texture_feats['glcm_contrast'],
            texture_feats['glcm_homogeneity'],
            texture_feats['glcm_energy'],
            texture_feats['glcm_entropy'],
            texture_feats['tamura_coarseness'],
            texture_feats['tamura_contrast'],
            texture_feats['tamura_directionality']
        ] + [ripeness]
        
        return np.array(features, dtype=np.float32)
        
        
    def prepare_data(self, data_dir):
        images = []
        traditional_features = []
        labels = []
        error_files = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for file in os.listdir(class_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                try:
                    file_path = os.path.join(class_path, file)
                    image = cv2.imread(file_path)
                    
                    if image is None:
                        error_files.append(file_path)
                        continue
                    
                    contrast_img, hsv_image, mask = preprocess_image(image)
                
                    color_feats = color_features(hsv_image, mask)
                    texture_feats = texture_features(contrast_img, mask)
                    ripeness = ripeness_factor(hsv_image, mask)
                    
                    combined_features = (
                        color_feats +
                        [
                            texture_feats['glcm_contrast'],
                            texture_feats['glcm_homogeneity'],
                            texture_feats['glcm_energy'],
                            texture_feats['glcm_entropy'],
                            texture_feats['tamura_coarseness'],
                            texture_feats['tamura_contrast'],
                            texture_feats['tamura_directionality']
                        ] +
                        [ripeness]
                    )
                    
                    images.append(contrast_img)
                    traditional_features.append(combined_features)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_files.append(file_path)
                    continue
        
        return np.array(images), np.array(traditional_features), np.array(labels)
        
    def train(self, train_images, train_traditional_features, train_labels, 
            epochs=70, batch_size=10):
        unique_labels, label_counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        class_weights = {i: total_samples / (len(unique_labels) * count) 
                        for i, count in zip(unique_labels, label_counts)}
        
        train_idx, val_idx = train_test_split(
            np.arange(len(train_images)), 
            test_size=0.2,
            random_state=42,
            stratify=train_labels 
        )
        
        x_train = [
            train_images[train_idx],
            train_traditional_features[train_idx]
        ]
        x_val = [
            train_images[val_idx],
            train_traditional_features[val_idx]
        ]
        y_train = train_labels[train_idx]
        y_val = train_labels[val_idx]
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            class_weight=class_weights  
        )
        
        return history

    def fine_tune(self, train_images, train_traditional_features, train_labels,
                epochs=70, batch_size=10):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.models.Model):
                for base_layer in layer.layers[100:]:
                    base_layer.trainable = True
               
        unique_labels, label_counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        class_weights = {i: total_samples / (len(unique_labels) * count) 
                        for i, count in zip(unique_labels, label_counts)}
                        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            [train_images, train_traditional_features],
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            class_weight=class_weights  
        )
        
        return history
        
    def evaluate(self, test_images, test_traditional_features, test_labels):
        predictions = []
        confidences = []
        
        for i in range(0, len(test_images), 32):
            batch_imgs = test_images[i:i+32]
            batch_features = test_traditional_features[i:i+32]
            
            pred = self.model.predict(
                [batch_imgs, batch_features],
                verbose=0
            )
            pred_classes = np.argmax(pred, axis=1)
            pred_confidence = np.max(pred, axis=1)
            
            predictions.extend(pred_classes)
            confidences.extend(pred_confidence)
            
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, predictions),
            'precision_macro': precision_score(test_labels, predictions, average='macro'),
            'recall_macro': recall_score(test_labels, predictions, average='macro'),
            'f1_macro': f1_score(test_labels, predictions, average='macro'),
            'precision_weighted': precision_score(test_labels, predictions, average='weighted'),
            'recall_weighted': recall_score(test_labels, predictions, average='weighted'),
            'f1_weighted': f1_score(test_labels, predictions, average='weighted')
        }
        
        for class_id in range(self.num_classes):
            class_pred = (predictions == class_id)
            class_true = (test_labels == class_id)
            metrics[f'precision_class_{class_id}'] = precision_score(class_true, class_pred)
            metrics[f'recall_class_{class_id}'] = recall_score(class_true, class_pred)
            metrics[f'f1_class_{class_id}'] = f1_score(class_true, class_pred)
         
            class_confidences = confidences[predictions == class_id]
            metrics[f'avg_confidence_class_{class_id}'] = (
                np.mean(class_confidences) if len(class_confidences) > 0 else 0
            )
        
        metrics['confusion_matrix'] = tf.math.confusion_matrix(
            test_labels,
            predictions,
            num_classes=self.num_classes
        ).numpy()
 
        print("\nEnhanced Evaluation Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        
        print("\nPer-class Performance:")
        class_names = ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']
        for class_id in range(self.num_classes):
            print(f"\nClass {class_names[class_id]}:")
            print(f"  Precision: {metrics[f'precision_class_{class_id}']:.4f}")
            print(f"  Recall: {metrics[f'recall_class_{class_id}']:.4f}")
            print(f"  F1-Score: {metrics[f'f1_class_{class_id}']:.4f}")
            print(f"  Avg Confidence: {metrics[f'avg_confidence_class_{class_id}']:.4f}")
        
        return metrics
    
    def visualize_results(self, metrics, save_path=None):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        plt.imshow(metrics['confusion_matrix'], cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        class_names = ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
       
        thresh = metrics['confusion_matrix'].max() / 2
        for i in range(metrics['confusion_matrix'].shape[0]):
            for j in range(metrics['confusion_matrix'].shape[1]):
                plt.text(j, i, format(metrics['confusion_matrix'][i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if metrics['confusion_matrix'][i, j] > thresh else "black")
        
        plt.subplot(122)
        class_metrics = {
            'Precision': [metrics[f'precision_class_{i}'] for i in range(self.num_classes)],
            'Recall': [metrics[f'recall_class_{i}'] for i in range(self.num_classes)],
            'F1-Score': [metrics[f'f1_class_{i}'] for i in range(self.num_classes)]
        }
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.bar(x - width, class_metrics['Precision'], width, label='Precision')
        plt.bar(x, class_metrics['Recall'], width, label='Recall')
        plt.bar(x + width, class_metrics['F1-Score'], width, label='F1-Score')
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-class Performance Metrics')
        plt.xticks(x, class_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path):
        self.model.save(path)

def train_hybrid_model():
    DATA_DIR = 'D:\Material\Postgraduate\Computer Version\Project\Image'
    
    classifier = HybridBananaClassifier()
    images, traditional_features, labels = classifier.prepare_data(DATA_DIR)
    
    train_images, test_images, train_trad, test_trad, train_labels, test_labels = \
        train_test_split(images, traditional_features, labels, 
                        test_size=0.3, random_state=42,
                        stratify=labels)  # Ensure balanced split
    

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    print("\nStarting pre-training with enhanced optimization:")
    history = classifier.train(train_images, train_trad, train_labels, epochs=20)
    
    print("\nStarting fine-tuning with advanced training strategy:")
    history_ft = classifier.fine_tune(train_images, train_trad, train_labels, epochs=50)
    
    metrics = classifier.evaluate(test_images, test_trad, test_labels)
    classifier.visualize_results(metrics, save_path='hybrid_evaluation_results.png')

    classifier.save_model('hybrid_banana_classifier.keras')

def predict_images():
    TEST_DIR = 'D:/Material/Postgraduate/Computer Version/Project/Test'
    MODEL_PATH = 'hybrid_banana_classifier.keras'
    
    classifier = HybridBananaClassifier()
    classifier.model.load_weights(MODEL_PATH)
    
    class_names = ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']
    
    for img_name in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img_name)
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {img_name}")
                continue
            
            contrast_img, hsv_image, mask = preprocess_image(image)
            processed_img = cv2.resize(contrast_img, classifier.img_size)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            trad_features = classifier.extract_traditional_features(image)
            trad_features = np.expand_dims(trad_features, axis=0)
            
            pred = classifier.model.predict(
                [processed_img, trad_features], 
                verbose=0
            )
            pred_class = np.argmax(pred[0])
            confidence = float(pred[0][pred_class])
            
            print(f"{img_name}: Predicted class - {class_names[pred_class]} "
                  f"(Confidence: {confidence:.2%})")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

if __name__ == '__main__':
    train_hybrid_model()
    predict_images()