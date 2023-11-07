import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from pathlib import Path
import os


st.set_page_config(
    page_title = 'EDA of Cat or Dog Prediction'
)

def run():
        
    # Membuat Title
    st.title ('Farhan\'s Cat or Dog Recognition Illustration')

    # Membuat Sub Header
    st.subheader ('EDA for Cat or Dog Analyst')

    # Menambahkan Gambar
    st.image("https://media.giphy.com/media/dC8jdwiSuBiet1SVgD/giphy-downsized-large.gif", 
             caption ='Cat and Dog')

    # Menambahkan Deskripsi
    st.write ('Page created by **Farhan** for Hacktiv8 Assignment')

    # Membuat Garis Lurus
    st.markdown ('---')


    # From notebook
    data_dir = "./cat_dog/"
    data_dir = "./cat_dog/"
    train_dir = data_dir + "./training_set"
    test_dir = data_dir + "./test_set/"

    train_path = Path(train_dir)
    test_path = Path(test_dir)

    train_img_path = list(train_path.glob(r'*/*.jpg'))
    train_img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],train_img_path))

    test_img_path = list(test_path.glob(r'*/*.jpg'))
    test_img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],test_img_path))

        # Data Train
    train_img_path_series = pd.Series(train_img_path,name="Image").astype(str)
    train_img_labels_series = pd.Series(train_img_labels,name="Category")

    df_train = pd.concat([train_img_path_series,train_img_labels_series],axis=1)

        # Data Test
    test_img_path_series = pd.Series(test_img_path,name="Image").astype(str)
    test_img_labels_series = pd.Series(test_img_labels,name="Category")

    df_test = pd.concat([test_img_path_series,test_img_labels_series],axis=1)

    # Show The Data
    st.write ('## The datasets for base prediction')

    # Distribusi data
    st.write ('### The class distribusion')

    distribusi_train = df_train['Category'].value_counts()
    count_df_train = pd.DataFrame({'Category': distribusi_train.index, 'Count': distribusi_train.values})

    distribusi_test = df_test['Category'].value_counts()
    count_df_test = pd.DataFrame({'Category': distribusi_test.index, 'Count': distribusi_test.values})

    count_df = count_df_train + count_df_test

    fig = px.pie(count_df, names='Category', values='Count', hover_data=['Count'],
                labels={'Category': 'Category', 'Count': 'Count'})

    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title_text='Distribution of Classes')

    st.plotly_chart(fig)

    st.write('As we can see, in the dataset there is more fire picture rather than no fire picture')
    st.write('For modeling purpose, we have to balance the image count')

    # Image Comparisson
    st.write ('### Image Comparison')
              
    cat_train_images = df_train[df_train['Category'] == 'cats']['Image'].head(5).tolist()
    dog_train_images = df_train[df_train['Category'] == 'dogs']['Image'].head(5).tolist()

    cat_test_images = df_test[df_test['Category'] == 'cats']['Image'].head(5).tolist()
    dog_test_images = df_test[df_test['Category'] == 'dogs']['Image'].head(5).tolist()

    def display_images(image_paths, title):
        fig, axis = plt.subplots(1, len(image_paths), figsize=(15, 4))
        fig.suptitle(title, fontsize=16)
        for i, img_path in enumerate(image_paths):
            img = plt.imread(img_path, format='PNG')
            axis[i].imshow(img)
            axis[i].set_title(f'{img.shape}')
            axis[i].axis('off')
        return fig

    cat_train_img = display_images(cat_train_images, 'Cats Images in Data Train')
    dog_train_img = display_images(dog_train_images, 'Dogs Images in Data Train')

    cat_test_img = display_images(cat_test_images, 'Cats Images in Data Test')
    dog_test_img = display_images(dog_test_images, 'Dogs Images in Data Test') 

    st.write ('The image for predicition that define as cat')
    st.pyplot(cat_train_img)
    st.pyplot(cat_test_img)
    st.write ('The image for predicition that define as dog')
    st.pyplot(dog_train_img)
    st.pyplot(dog_test_img)

    st.write('From the image above, we can see which is cat and dog')
    st.write('But the machine don\'t, so we have to make model that machine can know which is cat or dog')
 
    # Size gambar
    st.write('### Picture Size')

    
    cat = './cat_dog/training_set/cats/cat.301.jpg'
    img_cat = plt.imread(cat)
    fig, ax =  plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_cat)
    ax[0].axis('on')  
    ax[0].set_title('Cat Image')

    dog = './cat_dog/test_set/dogs/dog.4114.jpg'
    img_dog = plt.imread(dog)
    ax[1].imshow(img_dog)
    ax[1].axis('on')  
    ax[1].set_title('Dog Image')
    st.pyplot(fig)

    st.write('The size of the image is different, to do the analysis we have to change it to the same size to this')

    # Cat Resized
    img_cat_resized = Image.open(cat)
    desired_size = (150, 150)
    resized_img_cat = img_cat_resized.resize(desired_size)

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    ax2[0].imshow(resized_img_cat)
    ax2[0].axis('on')  
    ax2[0].set_title('Cat Image Resized')

    # Dog Resized
    img_dog_resized = Image.open(dog)
    resized_img_dog = img_dog_resized.resize(desired_size)

    ax2[1].imshow(resized_img_dog)
    ax2[1].axis('on')  
    ax2[1].set_title('Dog Image Resized')
    st.pyplot(fig2)

    st.write('If you didn\'t notice, check the image size and the quality of the image')

if __name__ == '__main__':
    run()