import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import re,requests,pymysql
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score

'''資料清單
v1.品牌名稱+照片網址 brand_photo
v2.手機照片網址 phone_photo
   手機完整名稱 phone_name
v3.匯率 rate
4.Xy變數集合 info_frame (品牌對應)
'''

def url_bs(url):
    res=requests.get(url)
    return BeautifulSoup(res.text,'lxml')
def connect_mysql(host='localhost', user='root', password='her888480'):
    global connect, cursor
    connect = pymysql.connect(
        host=host,
        user=user,
        password=password,
        charset='utf8mb4', # 指定資料庫連線的字符集
        use_unicode=True)  # 指定是否使用Unicode字串處理資料庫中的文字資料
    cursor = connect.cursor()
def predict_new_data(new_data): # 新數據定義
    new_data_encoded = te.transform(new_data)
    new_data_scaled = scaler.transform(new_data_encoded)
    # 歐元換算台幣
    prediction = round(rf.predict(new_data_scaled)[0]*rate)
    prediction = "{:,}".format(prediction)
    return prediction
def generate_template_data(nearest_data):
    template_data = {}
    for i in range(5):
        template_data["img%d" % (i+1)] = nearest_data["phone_photo"].iloc[i]
        template_data["phone_name%d" % (i+1)] = nearest_data["phone_name"].iloc[i]
        template_data["screen%d" % (i+1)] = nearest_data["screen"].iloc[i]
        template_data["battery%d" % (i+1)] = nearest_data["battery"].iloc[i]
        template_data["camera%d" % (i+1)] = nearest_data["camera"].iloc[i]
        template_data["ram%d" % (i+1)] = nearest_data["ram"].iloc[i]
        template_data["storage%d" % (i+1)] = nearest_data["storage"].iloc[i]
    return template_data

# # 擷取品牌照片網址
# bs=url_bs('https://zh.kalvo.com/brands/')
# box1=bs.find("section","sec1")
# con1=box1.find_all("a")
# con2=box1.find_all("img")
# brand_photo=[[i.get("alt"),i.get("data-src")] for i in con2]
# info_lst=[]

# # 變數蒐集
# for i,l in zip(con1,brand_photo): # 進入單個品牌網頁
#     brand_url=i.get("href")
#     try:
#         bs=url_bs(brand_url)
#         box2=bs.find("ul","table col1 list1")
#         if box2:
#             box3=box2.find_all("a","shine")
#             for j in box3: # 進入單個產品網頁
#                 if "Pad"not in j.get("title") and\
#                    "Tab" not in j.get("title") and\
#                    "HTC A" not in j.get("title") and\
#                    "Blackview Oscal Spider" not in j.get("title") and\
#                    "Allview Viva" not in j.get("title") and\
#                    "Alcatel 3T" not in j.get("title") and\
#                    "TCL Nxt" not in j.get("title") and\
#                    "Plum Optimax" not in j.get("title") and\
#                    "HMD T" not in j.get("title") and\
#                    not re.search("Doogee [O|T|U|R]", j.get("title")) and\
#                    not re.search('OUKITEL [R|O]', j.get("title")) and\
#                    brand_url!="https://zh.kalvo.com/brands/amazon/":
#                     phone_url=j.get("href")
#                     try:
#                         bs=url_bs(phone_url)
#                         # 手機名稱(品牌+型號)
#                         phone_name=bs.find("div","head relative").text.strip()
#                         # 定位X變數 (螢幕大小 (英寸)、電池容量 (mAh)、記憶體RAM (GB)、相機 (MP像素)、容量)
#                         box4=bs.find("div","body relative")
#                         con3=box4.find_all("strong")
#                         # 定位y應變數 (價格 €歐元)
#                         box5=bs.find("div","specTabels")
#                         con4=box5.find_all("td")
                        
#                         for k in con4: #擷取變數
#                             if "价格" in k.text:
#                                 if "€" in k.find_next_sibling("td").text:
                                    
#                                     if con3[1].text.strip()=="—":
#                                         screen=0
#                                     else:
#                                         screen=con3[1].text.strip()
#                                     if con3[2].text.strip()=="—":
#                                         battery=0
#                                     else:
#                                         battery=con3[2].text.strip()
#                                     if con3[3].text.strip()=="—":
#                                         ram=0
#                                     elif "-" in con3[3].text.strip():
#                                         ram=int(re.search(r"(\d+)-",con3[3].text.strip()).group(1))
#                                     elif "/" in con3[3].text.strip():
#                                        ram=int(re.search(r"(\d+)/",con3[3].text.strip()).group(1))
#                                     else:
#                                         ram=con3[3].text.strip()
#                                         ram=int(re.sub(r'[^0-9.]','',ram))
#                                     if con3[4].text.strip()=="—":
#                                         camera=0
#                                     else:
#                                         camera=con3[4].text.strip()
#                                         camera=float(re.sub(r'[^0-9.]','',camera))
#                                     if con3[5].text.strip()=="—":
#                                         storage=0
#                                     elif "-" in con3[5].text.strip():
#                                        storage=int(re.search(r"(\d+)[\sA-Z]*-",con3[5].text.strip()).group(1))
#                                     elif "/" in con3[5].text.strip():
#                                        storage=int(re.search(r"(\d+)[\sA-Z]*/",con3[5].text.strip()).group(1))
#                                     else:
#                                         storage=con3[5].text.strip()
#                                         storage=int(re.sub(r'[^0-9.]','',storage))
#                                     # 用正則表達式格式化價格字串
#                                     price=float(re.search(r'€\s*([\d,\.]+)',k.find_next_sibling("td").text).group(1).replace(",",""))
#                                     # re.sub移除千分位逗號、單位
#                                     info_lst.append([l[0],float(re.sub(r'[^0-9.]','',screen)),
#                                                      int(re.sub(r'[^0-9.]','',battery)),
#                                                      ram, camera, storage, price,
#                                                      phone_name,l[1],
#                                                      box4.find("img","lazyload").get("data-src")])# 擷取手機照片網址
#                                     break
#                     except Exception as e:
#                         print(f"<{phone_url}> Erro:{e}")
#     except Exception as e:
#         print(f"<{phone_url}> Erro:{e}")                    
# # CSV匯出
# info_frame=pd.DataFrame(info_lst,columns=["brand","screen","battery","ram","camera","storage","price",
#                                           "phone_name","brand_photo","phone_photo"])
# info_frame.to_csv("phone_data.csv",index=False)

# # 建立資料庫連線
# try:
#     connect_mysql()
#     # 建立新的database "topics"
#     cursor.execute("CREATE DATABASE IF NOT EXISTS `topics`")
#     connect.commit()
#     # 使用"topics"
#     cursor.execute("USE `topics`")
#     # 建立新table "phone_data"
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS `phone_data`(
#             brand TEXT,
#             screen FLOAT,
#             battery INT,
#             ram INT,
#             camera FLOAT,
#             storage INT,
#             price FLOAT,
#             phone_name TEXT,
#             brand_photo TEXT,
#             phone_photo TEXT
#             )""")
#     connect.commit()
#     # 清空 phone_data 表，以利更新資料
#     cursor.execute("TRUNCATE TABLE `phone_data`")
#     connect.commit()
#     # 插入資料到 phone_data 表
#     insert_query = """
#         INSERT INTO `phone_data` (brand,screen,battery,ram,camera,storage,price,phone_name,brand_photo,phone_photo)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
#     cursor.executemany(insert_query,info_frame.values.tolist())
#     connect.commit()
#     print("資料已成功插入")
# except pymysql.MySQLError as e:
#     print("連線錯誤:%s" % e)
# finally:
#     connect.close()

# # 連線資料庫讀取資料
# try:
#     connect_mysql()
#     cursor.execute("USE `topics`")
#     phone_data=pd.read_sql('SELECT*FROM `phone_data`',con=connect)#使用connect指定的Mysql獲取資料
# except pymysql.MySQLError as e:
#     print("連線錯誤:%s" % e)
# finally:
    connect.close()

phone_data=pd.read_csv("phone_data.csv")
# info_frame=phone_data
# info_lst=info_frame.values.tolist()
# phone_data.to_csv("phone_data.csv",index=False)

# 變數設定
X,y=phone_data.iloc[:,0:6],phone_data["price"]

# 目標編碼器 (指定brand欄位)
te=TargetEncoder(cols=['brand'])

# 交叉驗證 (處理有可能因目標編碼而產生資料泄漏的問題，避免模型被高估)
kf=KFold(n_splits=5, shuffle=True, random_state=0)
 # n_splits：交叉驗證的次數
 # shuffle：每次劃分前，資料集會被隨機打亂順序，有助於減少因數據順序性造成的偏差

y_oof = np.zeros(len(X))
for train_idx,valid_idx in kf.split(X):
    X_train,X_valid=X.iloc[train_idx],X.iloc[valid_idx]
    y_train,y_valid=y.iloc[train_idx],y.iloc[valid_idx]
    # 目標編碼器
    X_train=te.fit_transform(X_train,y_train)
    X_valid=te.transform(X_valid)
    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    # 隨機森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    # 預測驗證集
    y_oof[valid_idx] = rf.predict(X_valid)

# 訓練完整模型
X_te = te.fit_transform(X, y)
X_te = scaler.fit_transform(X_te)
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_te, y)
# # 特徵重要性
# feature_imp = pd.Series(rf.feature_importances_, index=phone_data.iloc[:, 0:6].columns).sort_values(ascending=False)
# print("Feature Importances:\n%s"%feature_imp)
# # 模型評分
# rmse = np.sqrt(mse(y, y_oof))
# r2 = r2_score(y, y_oof)
# print("Overall RMSE:%f"%rmse)
# print("Overall R-squared:%f"%r2)
# # 視覺化 (實際值vs.預測值) 散點圖
# plt.figure(figsize=(10,5))
# plt.scatter(y, y_oof)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')
# plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--') # 參考線
# plt.show()

# 擷取台灣銀行即期歐元匯率
bs=url_bs('https://rate.bot.com.tw/xrt?Lang=zh-TW')
rbox=bs.find_all("tr")[16]
if "歐元" in rbox.find("td","currency phone-small-font").text:
    rate=float(rbox.find("td",{"data-table":"本行即期買入"}).text.strip())
else:
    print("找不到匯率")

# 推薦數據建模
nn = NearestNeighbors(n_neighbors=5).fit(X.drop('brand', axis=1))

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def data():
    if request.method=="POST":
        try:
            brand_get=request.form["brand"]
            screen_get=request.form["screen"]
            battery_get=request.form["battery"]
            ram_get=request.form["ram"]
            camera_get=request.form["camera"]
            storage_get=request.form["storage"]
            new_data=pd.DataFrame({
                 'brand': [brand_get],
                 'screen': [screen_get],
                 'battery': [battery_get],
                 'ram': [ram_get],
                 'camera': [camera_get],
                 'storage': [storage_get]})
            prediction = predict_new_data(new_data)
            distances, indices = nn.kneighbors(new_data.drop('brand', axis=1))
            nearest_data = phone_data.iloc[indices[0]] # 推薦的五筆資料
            return render_template("test_data.html",
                                   prediction="$"+prediction ,
                                   **generate_template_data(nearest_data))
        except Exception as e:
            return str(e)
    return render_template("test_data.html")

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True      
    app.jinja_env.auto_reload = True
    app.run()
    