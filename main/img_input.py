# 入力画像のアップロード
import pathlib
input_filename = input('画像のパスを入力してください')

input_image = image.load_img(input_filename, target_size=(224,224))
#print(image_path_list)


# 行列形式への変換
import numpy as np
from tensorflow.keras.preprocessing import image

images = []

for img_path in image_path_list:
    img = image.load_img(img_path, target_size=(224, 224)) # 画像の読み込み
    raw_image = image.img_to_array(img) # 多次元配列への変換
    images.append(raw_image) # 変換したデータをimagesに追加
    
images = np.array(images) # 四次元のndarrayに変換


# ResNetの初期化
import tensorflow as tf
from tensorflow.keras.applications import resnet

model = tf.keras.applications.ResNet152(include_top=False,
                                       weights='imagenet',
                                       input_tensor=None,
                                       pooling='avg',
                                       classes=1000)


# 特徴ベクトルの作成
preprocessed = resnet.preprocess_input(images)
features = model.predict(preprocessed)
print(features)
features[0]