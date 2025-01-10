from model import BananaRipenessClassifier
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 创建分类器
    classifier = BananaRipenessClassifier()
    
    # 数据集路径
    data_dir = "D:/Material/Postgraduate/Computer Version/Project/Initial Image/Image"
    
    # 准备数据集
    print("Preparing dataset...")
    X, y = classifier.prepare_dataset(data_dir)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    print("Training model...")
    history = classifier.train(X_train, y_train)
    
    # 评估模型
    print("Evaluating model...")
    loss, accuracy = classifier.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 测试单张图片
    class_names = ['Green', 'Yellowish_Green', 'Midripen', 'Overripen']
    test_image_path = "D:/Material/Postgraduate/Computer Version/Project/Test/test.jpg"
    test_image = cv2.imread(test_image_path)
    
    if test_image is not None:
        prediction = classifier.predict(test_image)
        print(f"Predicted class: {class_names[prediction]}")

if __name__ == "__main__":
    main()