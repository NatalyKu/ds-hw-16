import streamlit as st
from PIL import Image, ImageOps
import io
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.losses import SparseCategoricalCrossentropy
import pickle

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("передбачення класу зображення")

st.write("""веб-застосунок, який дозволяє завантажувати зображення для класифікації за допомогою""")
st.write("""    - навченої згорткової нейронної мережі""")
st.write("""    - VGG16 нейронної мережі""")
st.write("""Відображує вхідне зображення , графіки функції втрат і точності для моделі; результати класифікації (ймовірності для кожного класу та передбачений клас) """)


def predict_image(img, model, model_type):
    if model_type == 'CNN':
        img = img.resize((28, 28))  # change img to 28x28
        img = img.convert('L')  # konvert to grayscale
        img = np.array(img) / 255.0  # normalize image
        img = np.expand_dims(img, axis=-1)  # Leggtil kanalakse
    elif model_type == 'VGG16':
        img = img.resize((224, 224))  # change img to 224x224
        img = np.array(img) / 255.0  # normalize image
        if img.shape[-1] == 1:  # Hvis bildet er gråskala, konverter til RGB
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[-1] == 4:  # Hvis bildet har en alfakanal, fjern den
            img = img[..., :3]
    else:
        raise ValueError("Ukjent model_type. Bruk 'CNN' eller 'VGG16'.")
    
    img = np.expand_dims(img, axis=0)  # Legg til batch-akse
    
    # Gjør prediksjonen
    prediction = model.predict(img)
    c_class = np.argmax(prediction, axis=1)[0]
    return prediction[0], c_class


def loss_and_accuracy_plot(history, xlim=None, ylim_loss=None, ylim_acc=None, yscale_loss='linear', yscale_acc='linear'):
    plt.figure(figsize=(12, 6))
    
    
    # Loss plot 
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    if xlim:
        plt.xlim(xlim)
    if ylim_loss:
        plt.ylim(ylim_loss)
    plt.yscale(yscale_loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    if xlim:
        plt.xlim(xlim)
    if ylim_acc:
        plt.ylim(ylim_acc)
    plt.yscale(yscale_acc)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    return plt

# Завантаження зображення
uploaded_file = st.file_uploader("# Виберіть зображення для класифікації:", type=["jpg", "jpeg", "png"])

model_choice = st.sidebar.selectbox('Chose model:', ['CNN', 'VGG16'])
if model_choice == 'CNN':
    model = load_model('CNN_model.keras') # pretrained CNN model
    with open('./CNN_history.pkl', 'rb') as f:
        history_obj = pickle.load(f)
        if isinstance(history_obj, dict):
            history = history_obj
        else:
            history = history_obj.history
    model_type = 'CNN'
else:
    model = load_model('VGG16_model.keras') # pretrained VGG16 model
    with open('./VGG16_history.pkl', 'rb') as f:
        history_obj = pickle.load(f)
        history = history_obj.history # Access the history attribute
    model_type = 'VGG16'

if uploaded_file is not None:
    # Відкриття зображення за допомогою Pillow
    image = Image.open(uploaded_file)
    st.image(image)
    if st.button('Make prediction'):
        progress = st.progress(0)

        for i in range(5):
            # Оновлення progress bar кожну секунду
            time.sleep(0.1)  # Пауза на 0.1 секунди
            progress.progress((i + 1) * 20)
        
        if model is not None:
            st.write("Making prediction...")

            probabilities, predicted_class = predict_image(image, model, model_type)
            st.write(f'Predicted class is: {class_names[predicted_class]}')
            graff = loss_and_accuracy_plot(history)
            st.pyplot(graff)
            st.write(f'Probability {class_names[0]}: {probabilities[0]:.2f}')
            st.write(f'Probability {class_names[1]}: {probabilities[1]:.2f}')
            st.write(f'Probability {class_names[2]}: {probabilities[2]:.2f}')
            st.write(f'Probability {class_names[3]}: {probabilities[3]:.2f}')
            st.write(f'Probability {class_names[4]}: {probabilities[4]:.2f}')
            st.write(f'Probability {class_names[5]}: {probabilities[5]:.2f}')
            st.write(f'Probability {class_names[6]}: {probabilities[6]:.2f}')
            st.write(f'Probability {class_names[7]}: {probabilities[7]:.2f}')
            st.write(f'Probability {class_names[8]}: {probabilities[8]:.2f}')
            st.write(f'Probability {class_names[9]}: {probabilities[9]:.2f}')
            
        else:
            st.write(f"{model_choice} model is not loaded. Please load the {model_choice} model to make predictions.")