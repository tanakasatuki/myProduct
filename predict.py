import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet
import tensorflow as tf

# 検索ロジックの作成
def cos_sim(v1, v2): # コサイン類似度を計算する関数を定義
         return np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))

def get_top_n_indexes(array, num): # 類似度が高い順にインデックスを取得する関数を定義
        idx = np.argpartition(array, -num)[-num:]
        return idx[np.argsort(array[idx])][::-1]

def search(query_vector, features, num):
    sims = []
    for vector in features:
            sim = cos(query_vector, vector)  # コサイン類似度を計算
            sims.append(sim)
    sims = np.array(sims)
    indexes = get_top_n_indexes(sims, num)  
    return indexes, sims[indexes] # num番目まで、値が大きい順にインデックス番号のリストindexとその類似度のリストsimsを返す


def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) # 画像の読み込み
    raw_image = image.img_to_array(img) # 多次元配列への変換

    images = [raw_image]
    images = np.array(images) # 四次元のndarrayに変換

    model = tf.keras.applications.ResNet152(include_top=False,
                                       weights='imagenet',
                                       input_tensor=None,
                                       pooling='avg',
                                       classes=1000)

    preprocessed = resnet.preprocess_input(images)
    feature = model.predict(preprocessed)

    features = np.load('./main/np_save.npy')
        
    # 実際に検索
    results, sims = search(feature[0], features, 1)

    #CSVの読み込み
    df = pd.read_csv('./dataflame/data_a.csv')
    for n in results:
      result_a = df.loc[n]

    return result_a