#Building the predictive system  for the algorithm

input_data=[9.9,11.3,56,77,89.6,23,55.6,67,89,114.114,67.00,78,0.8,0.7,0.55,55,66,77,88.77,123,45,78,90,1,2,3,4,5,6,7]


input_data_as_numpy_array=np.asarray(input_data)
df_reshape=input_data_as_numpy_array.reshape(1,-1)

df_std=scaler.transform(df_reshape)


prediction= model.predict(df_std)
print(prediction)


prediction_label=[np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0]==0):
    print("The tumor is malignant")
    
else:
    print("The tumor is Benign")