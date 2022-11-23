#importing required modules
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#reading the dataset
dataset = pd.read_csv('SOCR-HeightWeight.csv')
#setting x and y values
y=dataset['Weight(Pounds)']
x=dataset['Height(Inches)']

#creating model that is an empty brain
model=Sequential()
model.get_config()

#adding layers
model.add(
    Dense(
        units=1,
        activation="linear", 
        kernel_initializer="zeros",
        bias_initializer="zeros",
        input_shape=(1,) 
    )
)

#getting information about model summary
model.summary()

model.get_config()

#model.get_layer("dense").input
#model.get_layer("dense").output

model.compile(loss="mean_squared_error",optimizer="Adam")

#trianing model
model.fit(x, y,epochs=5)

w=model.predict([68.22])
print("Estimated Weight of that person is = ",w)
