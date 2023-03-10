import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet

# インストールtensorflow、pathlib、numpy、pandas、image、resnet、Pillow

#************************************画像のロード***********************************************
dir_path = "./images/"

image_path_list = []
for img_path in pathlib.Path(dir_path).glob("*.jpg"): # jpgの画像ファイルのパスをすべて取得
    image_path_list.append(img_path) # img_path_listにパスを追加
image_path_list.sort() # ファイル名を辞書順に並べる
#print(image_path_list)

#************************************行列形式への変換*******************************************
images = []

for img_path in image_path_list:
    img = image.load_img(img_path, target_size=(224, 224)) # 画像の読み込み
    raw_image = image.img_to_array(img) # 多次元配列への変換
    images.append(raw_image) # 変換したデータをimagesに追加
    
images = np.array(images) # 四次元のndarrayに変換

#*************************************ResNetの初期化*******************************************
model = tf.keras.applications.ResNet152(include_top=False,
                                       weights='imagenet',
                                       input_tensor=None,
                                       pooling='avg',
                                       classes=1000)


#**************************************特徴ベクトルの作成**************************************
preprocessed = resnet.preprocess_input(images)
features = model.predict(preprocessed)
#print(features)

#************************************** 特徴ベクトルの保存**************************************
np.save('./main/np_save', features)