import requests
import pandas as pd
import time

# urlから画像をダウンロードする関数を作成
def download_file(url, file_name):
    response = requests.get(url)
    image = response.content
    
    with open(file_name, "wb") as f:
        f.write(image)

df = pd.read_csv('./data_b.csv') # csvファイルの読み込み
 # cafe_images1列だけを読み込んてリストにする


url_list = list(df['画像'])

for i in range(len(url_list)):
    url = url_list[i]
    file_name = './images/' + '{0:04d}'.format(272 + i) + '.jpg' #imagesフォルダに順番に保存
    download_file(url, file_name)
    print(i) #何件目までダウンロードできたか出力
    time.sleep(1) #サーバーに負荷をかけないように1秒スリープ
