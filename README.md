You can clone this repository by using the following Git command:  
`git clone https://github.com/Linyun-Gong/Banana-Ripeness-Classifier.git`

## Reference Paper
1. Mazen, F.M.A., Nashat, A.A. "Ripeness Classification of Bananas Using an Artificial Neural Network." Arab J Sci Eng 44, 6901–6910 (2019). https://doi.org/10.1007/s13369-018-03695-5
2. R. E. Saragih and A. W. R. Emanuel, "Banana Ripeness Classification Based on Deep Learning using Convolutional Neural Network," 2021 3rd East Indonesia Conference on Computer and Information Technology (EIConCIT), Surabaya, Indonesia, 2021, pp. 85-89, doi: 10.1109/EIConCIT50028.2021.9431928

## Dataset
1. From Paper-1
2. https://www.kaggle.com/datasets/lucianon/banana-ripeness-dataset
3. Train Data in the folder: Image
   Including 4 categories: Green / Yellowish Green / Midripen / Overrippen
4. Test Data in the folder: Test

## Implement
### ANN
Implementation based on Paper 1  
Trained Model: best_model.keras  
Difference: Otsu's thresholding method is also used when implementing brown spot detection   

### CNN Model  
Implementation based on Paper 2  
Trained Model: banana_ripeness_classifier.keras (MobileNetV2)   
banana_ripeness_classifier_2.keras (NASNetMobile)  

1. Train the model and predict:  
 `cd CNN_Model `  
`python CNN_model.py `  
2. Only predict (Use existing Model):  
Comments` train_model() `in CNN_model.py
3. Change model type: 
Replace model_type='mobilenet' to model_type='nasnetmobile' in CNN_model.py

### Hybrid CNN Model
Have two feature braches:  
1. Traditional Feature Branches
- 9 Color Features:  
For the three HSV channels (Hue, Saturation, Value): Mean value, Standard Deviation, Median  
- 7 Texture Features:  
GLCM: Contrast, Homogeneity, Energy, Entropy / Tamura: Coarseness, Contrast, Directionality
- Ripeness Factor
2. Image branching (CNN models)

Trained Model: hybrid_banana_classifier.keras

1. Train the model and predict:  
 `cd Hybrid_CNN `  
`python hybrid_CNN_model.py `  
2. Only predict (Use existing Model):  
Comments` train_model() `in hybrid_CNN_model.py
  
