import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2, NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class BananaRipenessClassifier:
   def __init__(self, model_type='mobilenet'):
       self.img_size = (224, 224)
       self.num_classes = 4
       self.model_type = model_type
       self.model = self._build_model()
       
   def _build_model(self):
       if self.model_type == 'mobilenet':
           base_model = MobileNetV2(
               weights='imagenet',
               include_top=False,
               input_shape=(*self.img_size, 3)
           )
       else:
           base_model = NASNetMobile(
               weights='imagenet', 
               include_top=False,
               input_shape=(*self.img_size, 3)
           )
           
       x = base_model.output
       x = GlobalAveragePooling2D()(x)
       x = Dropout(0.3)(x)
       predictions = Dense(self.num_classes, activation='softmax')(x)
       
       model = Model(inputs=base_model.input, outputs=predictions)
       
       for layer in base_model.layers:
           layer.trainable = False
           
       return model
       
   def preprocess_image(self, image):
       filtered = cv2.bilateralFilter(image, 9, 75, 75)
       resized = cv2.resize(filtered, self.img_size)
       normalized = resized / 255.0
       return normalized
   
   def prepare_data(self, data_dir):
       images = []
       labels = []
       error_files = []
       
       try:
           class_names = sorted(os.listdir(data_dir))
       except Exception as e:
           print(f"Error reading directory {data_dir}: {str(e)}")
           return None, None
           
       for class_idx, class_name in enumerate(class_names):
           class_dir = os.path.join(data_dir, class_name)
           print(f"Processing class: {class_name}")
           
           try:
               file_names = os.listdir(class_dir)
           except Exception as e:
               print(f"Error reading directory {class_dir}: {str(e)}")
               continue
               
           for img_name in file_names:
               try:
                   img_path = os.path.abspath(os.path.join(class_dir, img_name))
                   if not os.path.exists(img_path):
                       print(f"File not found: {img_path}")
                       error_files.append(img_path)
                       continue
                       
                   try:
                       image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                   except Exception as e:
                       print(f"Error decoding image {img_path}: {str(e)}")
                       error_files.append(img_path)
                       continue
                       
                   if image is None:
                       print(f"Failed to load image: {img_path}")
                       error_files.append(img_path)
                       continue
                       
                   processed_img = self.preprocess_image(image)
                   images.append(processed_img)
                   labels.append(class_idx)
                   
               except Exception as e:
                   print(f"Error processing {img_path}: {str(e)}")
                   error_files.append(img_path)
                   continue
       
       if error_files:
           print("\nFailed to load following files:")
           for file in error_files:
               print(file)
               
       if not images:
           print("No images were successfully loaded!")
           return None, None
           
       print(f"\nSuccessfully loaded {len(images)} images")
       return np.array(images), np.array(labels)
   
   def train(self, train_images, train_labels, epochs=50, batch_size=10):
       train_idx, val_idx = train_test_split(
           np.arange(len(train_images)), 
           test_size=0.2, 
           random_state=42
       )
       
       x_train, x_val = train_images[train_idx], train_images[val_idx]
       y_train, y_val = train_labels[train_idx], train_labels[val_idx]
       
       tf.keras.mixed_precision.set_global_policy('mixed_float16')
       
       datagen = ImageDataGenerator(
           rotation_range=20,
           width_shift_range=0.2,
           height_shift_range=0.2,
           horizontal_flip=True,
           vertical_flip=True,
           brightness_range=[0.8,1.2],
           zoom_range=0.2,
           shear_range=0.2
       )
       
       self.model.compile(
           optimizer='adam',
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy']
       )
       
       train_generator = datagen.flow(
           x_train, 
           y_train,
           batch_size=batch_size
       )
       
       val_generator = ImageDataGenerator().flow(
           x_val,
           y_val,
           batch_size=batch_size
       )
       
       history = self.model.fit(
           train_generator,
           epochs=epochs,
           validation_data=val_generator,
           steps_per_epoch=len(x_train) // batch_size,
           validation_steps=len(x_val) // batch_size
       )
       
       return history
   
   def fine_tune(self, train_images, train_labels, epochs=50, batch_size=10):
       if self.model_type == 'mobilenet':
           for layer in self.model.layers[100:]:
               layer.trainable = True
       else:
           for layer in self.model.layers[600:]:
               layer.trainable = True
               
       self.model.compile(
           optimizer=tf.keras.optimizers.Adam(1e-5),
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy']
       )
       
       history = self.model.fit(
           train_images, train_labels,
           batch_size=batch_size,
           epochs=epochs,
           validation_split=0.2
       )
       
       return history
   
   def predict(self, image):
       processed = self.preprocess_image(image)
       processed = np.expand_dims(processed, axis=0)
       prediction = self.model.predict(processed)
       return np.argmax(prediction[0])
       
   def evaluate(self, test_images, test_labels):
       predictions = []
       for image in test_images:
           processed = np.expand_dims(image, axis=0)
           pred = self.model.predict(processed, verbose=0)
           pred_class = np.argmax(pred[0])
           predictions.append(pred_class)
           
       predictions = np.array(predictions)
       
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
       
       metrics['confusion_matrix'] = tf.math.confusion_matrix(
           test_labels, 
           predictions,
           num_classes=self.num_classes
       ).numpy()
       
       print("\nEvaluation Results:")
       print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
       print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
       print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
       print("\nPer-class Performance:")
       for class_id in range(self.num_classes):
           print(f"Class {class_id}:")
           print(f"  Precision: {metrics[f'precision_class_{class_id}']:.4f}")
           print(f"  Recall: {metrics[f'recall_class_{class_id}']:.4f}")
           print(f"  F1-Score: {metrics[f'f1_class_{class_id}']:.4f}")
       
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
           plt.savefig(save_path)
       plt.show()
   
   def save_model(self, path):
       self.model.save(path)

def train_model():
    DATA_DIR = 'D:/Material/Postgraduate/Computer Version/Project/Image'
    
    classifier = BananaRipenessClassifier(model_type='mobilenet')
    # classifier = BananaRipenessClassifier(model_type='nasnetmobile')
    images, labels = classifier.prepare_data(DATA_DIR)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42
    )
    
    print("\nStarting pre-training:")
    history = classifier.train(train_images, train_labels, epochs=20)
    
    print("\nStarting fine-tuning:")  
    history_ft = classifier.fine_tune(train_images, train_labels, epochs=50)
    
    metrics = classifier.evaluate(test_images, test_labels)
    classifier.visualize_results(metrics, save_path='evaluation_results.png')
    
    classifier.save_model('banana_ripeness_classifier.keras')

def predict_images():
    TEST_DIR = 'D:/Material/Postgraduate/Computer Version/Project/Test'
    MODEL_PATH = 'banana_ripeness_classifier.keras'
    
    classifier = BananaRipenessClassifier(model_type='mobilenet')
    # classifier = BananaRipenessClassifier(model_type='nasnetmobile')
    classifier.model.load_weights(MODEL_PATH)
    
    class_names = ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']
    
    for img_name in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to load {img_name}")
            continue
            
        pred_class = classifier.predict(image)
        print(f"{img_name}: Predicted class - {class_names[pred_class]}")

if __name__ == '__main__':
    train_model()
    predict_images()