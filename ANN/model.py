import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 添加这行
import numpy as np
from features import extract_features
import cv2
import os

class BananaRipenessClassifier:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        """构建ANN模型"""
        model = models.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(10, activation='sigmoid'),
            layers.Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def augment_data(self, image, max_images=5):
        augmented = []
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        image = np.expand_dims(image, 0)
        it = datagen.flow(image, batch_size=1)
        for _ in range(max_images):
            augmented.append(next(it)[0])
        
        return augmented
    
    def prepare_dataset(self, data_dir):
        """准备数据集"""
        features = []
        labels = []
        
        class_names = ['Green', 'Yellowish_Green', 'Midripen', 'Overripen']
        
        for class_id, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            
            print(f"Processing class: {class_name}")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # 原始图像特征
                feature_vector = extract_features(image)
                features.append(feature_vector)
                labels.append(class_id)
                
                # 数据增强
                augmented_images = self.augment_data(image)
                for aug_img in augmented_images:
                    aug_feature = extract_features(aug_img)
                    features.append(aug_feature)
                    labels.append(class_id)
        
        return np.array(features), np.array(labels)
    
    def train(self, X_train, y_train, epochs=50, batch_size=10, validation_split=0.3):
        """训练模型"""
        # 转换为one-hot编码
        y_train = tf.keras.utils.to_categorical(y_train)
        
        # 添加早停机制
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 模型检查点
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # 训练
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint]
        )
        
        return history
    
    def predict(self, image):
        """预测图像类别"""
        feature_vector = extract_features(image)
        feature_vector = np.expand_dims(feature_vector, 0)
        prediction = self.model.predict(feature_vector)
        return np.argmax(prediction[0])
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_test = tf.keras.utils.to_categorical(y_test)
        return self.model.evaluate(X_test, y_test)