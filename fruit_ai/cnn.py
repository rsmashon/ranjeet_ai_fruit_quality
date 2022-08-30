from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
import keras

classifier = Sequential()


classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3),padding='same', activation = 'relu'))                       

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64,(3,3),padding='same', activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128,(3,3),padding='same', activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())

 

classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))
classifier.add(Dense(units=128,activation = 'relu'))


classifier.add(Dense(units=10,activation = 'softmax'))
  


classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True,
                                   vertical_flip= True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2
                                   ) 

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                target_size =(64,64),
                                                color_mode='rgb',
                                                batch_size=16,
                                                class_mode='categorical',
                                                shuffle=True
                                                  )

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),  
                                            batch_size=16,
                                            color_mode='rgb',
                                            class_mode='categorical')

checkpoint= keras.callbacks.ModelCheckpoint(r'C:\Users\Randhawa\Desktop\fruit_ai\models\fruit_quality_model.h5',
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)

callbacks=checkpoint
                                            

classifier.fit_generator(training_set,
                         steps_per_epoch=training_set.samples//16,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps=test_set.samples//16,
                         callbacks=callbacks
                         ) 

classifier.save("fruit_quality_model.h5")