#verbose is the progress tracker: 1 - progress bar, 2 - text
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import callbacks

def Directory_Selector(prompt, title):
    '''
    input: 2 strings: one that is printed, the other that is the title of a file explorer window
    output: The path of a directory, chosen by the user
    '''
    print(prompt) #prints a specific sentence
    Directory_Path = filedialog.askdirectory(title=title) #opens a file explorer window for the user to choose a directory
    return Directory_Path #returns the chosen directory's path

def Data_Preprocessing(data_dir, img_height, img_width,train_dir, validation_dir,batch_size):
    '''
    input: data_dir = path of directory which contains 2 sub-directories for train and validation,
    img_height, img_width = the wanted size of the images,
    train_dir, validation_dir = the 2 sub-directories, each containing 2 sub-directories for glasses and no-glasses images
    batch_size = the batch size that we'll use
    output: 2 'Generator' objects for train and validation that we'll use in the training of the model
    '''
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    validation_datagen=ImageDataGenerator(rescale=1./255)
#creating a generator for train data
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=(img_height, img_width),  # all images will be resized
        batch_size=batch_size,
        class_mode='binary'  # since we use binary_crossentropy loss, we need binary labels
        )

# this is a similar generator, for validation data
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary'
            )
    return train_generator, validation_generator
    
def SampleCounter(train_dir, validation_dir):
    '''
    Input: path of train and validation directories
    Output: number of images in train and in validation (seperate variables)
    '''
    #putting in variables the directory paths for train class-subfolders
    glasses_train_dir=os.path.join(train_dir, 'glasses')
    no_glasses_train_dir=os.path.join(validation_dir,'no_glasses')
    #putting in variables the directory paths for validation class-subfolders 
    glasses_valid_dir=os.path.join(validation_dir, 'glasses')
    no_glasses_valid_dir=os.path.join(validation_dir, 'no_glasses')
    #setting variables for total amount of images in train and in validation
    train_samples= len(os.listdir(glasses_train_dir)) + len(os.listdir(no_glasses_train_dir))
    validation_samples= len(os.listdir(glasses_valid_dir)) + len(os.listdir(no_glasses_valid_dir))
    return train_samples, validation_samples

def Model_Creation(img_height, img_width):
    '''
    Input: Height and Width of our images (after resize)
    Output: The model (Object keras.Model)
    '''
    #checking the format of the images, for setting a correct input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width, 3)
    #Creating the actual model (structure)
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', #We will use the loss function "Binary Cross Entropy to evaluate the model.
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print (model.summary()) #print a summary of the model we created (structure, shape, details)
    return model

def Model_Training(model,train_generator, train_samples,batch_size,epochs,validation_generator, validation_samples):
    '''
    Input: the model (Object keras.Model)
    Output: A trained model, and the history - for plotting purposes
    '''
    earlyStopping=callbacks.EarlyStopping(patience=2) #Model Training involves Early Stopping - a method that prevents overfitting
    history= model.fit_generator(train_generator,
        steps_per_epoch = train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    verbose=2, callbacks=[earlyStopping])
    return model, history

def Image_Selector(prompt, title):
    '''
    input: 2 strings: one that is printed, the other that is the title of a file explorer window
    output: the path of an image, chosen by the user
    '''
    print(prompt)
    filepath = filedialog.askopenfilename(title= title,
                                          multiple=False,
                                          filetypes = (("png images", '*.png'), ("jpeg images", "*.jpg"))
                                          )
    return filepath

def Prediction(model, img_height, img_width):
    '''
    input: the model, image height, image width
    output: a window with the selected image and the class of that image (predicted by the model)
    '''
    prompt = "Choose an image to test" #the string that is printed for the user
    title = "Select an Image (PNG/JPEG ONLY)" #the title of the file explorer tab
    img_path = Image_Selector(prompt, title) #The Choosing of the File
    img= image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) #convert the image to an array
    img_array /= 255.0 #the model can only work with pixel values between 0 and 1, so we rescale
    img_batch = np.expand_dims(img_array, axis=0)
    plt.imshow(img) 
    prediction= model.predict_classes(img_batch)
    if (prediction==0): #predction = 0 means that the class was 'Glasses'
        plt.title('Glasses') #title the image 'Glasses'
    else:
        plt.title('No Glasses') #title the image 'No Glasses'
    plt.show() #shows the image with the correct title
    return

def Plot_Accuracy(history):
    '''
    input: history of the model's training process
    output: 2 Graphs of the Accuracy: one is the train, one is the validation
    '''
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy') #the title of the graph
    plt.ylabel('accuracy') #the 'Y in the graph
    plt.xlabel('epoch') #the 'X' in the graph
    plt.legend(['train', 'validation'], loc='upper left') #naming the graphs and displaying it
    print("a graph of model's Accuracy")
    plt.show() #display
    return

def Plot_Loss(history):
    '''
    input: history of the model's training process
    output: 2 Graphs of the Loss: one is the train, one is the validation
    '''
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss') #the title of the graph
    plt.ylabel('loss') #the 'Y' in the graph
    plt.xlabel('epoch') #the 'X' in the graph
    plt.legend(['train', 'validation'], loc='upper left') #naming the graphs and displaying it
    print("a graph of model's loss")
    plt.show() #display
    return

def main():
    warnings.filterwarnings('ignore')
    #Part 1
    #setting core parameters    
    img_height , img_width = 200, 200
    batch_size=32
    epochs=10
    
    #setting up directories
    main_dir= Directory_Selector("Please Select the folder in which the project is located ('Glass_project' folder)", "Select the Directory Glass_Project, Path in English Only")
    data_dir= os.path.join(main_dir, 'faces')
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    
    train_generator, validation_generator = Data_Preprocessing(data_dir, img_height, img_width, train_dir, validation_dir,batch_size)
    
    #Part 2
    train_samples, validation_samples = SampleCounter(train_dir, validation_dir)
    model = Model_Creation(img_height, img_width)
    
    #The process of training, user can choose to train the model himself or use a pre-trained model
    #The only difference is the time it takes
    load_answer=input("Do you wish to load a pre-trained model, and skip the training process? (y=yes/n=no) ")
    while load_answer != 'y' and load_answer != 'n' :
        load_answer=input("Do you wish to load a pre-trained model , and skip the training process? (y=yes/n=no) ")
    if load_answer == 'y': #user chooses pre-trained model
        model.load_weights(os.path.join(main_dir, 'saved_weights.h5')) #loading the pre-trained model
        history=np.load(os.path.join(main_dir,'saved_history.npy'), allow_pickle='TRUE').item() #get the training history
    if load_answer == 'n': #user chooses to train the model himself
        model, history = Model_Training(model,train_generator,train_samples,batch_size,epochs,validation_generator,validation_samples) #Model training
        model.save_weights(os.path.join(main_dir, 'saved_weights.h5')) #Save the trained model weights
        np.save(os.path.join(main_dir,'saved_history.npy'), history.history) #Save the history of training
        history=history.history
    
    #Part 3
    #The process of predicting images, user can choose to predict more and more images, or to end
    flag=True #for the end loop
    img_answer = input("Do you wish to make a prediction? (y=yes/n=no) ")
    while (flag==True):
        while img_answer != 'y' and img_answer != 'n':   #User must input correctly 
            img_answer = input("Do you wish to make a prediction? (y=yes/n=no) ")
        if img_answer == 'y': #user chooses to make a Prediction
            Prediction(model, img_height, img_width) #The Prediction process
            img_answer = 'a' #random character that would activate the choice again
        if img_answer == 'n': #user chooses to end
            flag=False #end the loop
    ''' #optional, if you want to show 2 images of graphs of the loss and accuracy throughout the model's training
    Plot_Accuracy(history)
    Plot_Loss(history)
    '''

if __name__=='__main__':
    main()