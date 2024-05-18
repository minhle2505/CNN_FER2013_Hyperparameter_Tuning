#%%
token = "6747608024:AAEow8f3FsiwCFloYftRT9beziDLYtYfgqY"
#%%
import requests
#from token_bot import token
url = "https://api.telegram.org/bot/".replace("bot", "bot"+token)

resp = requests.get(url+'getMe')
print("staus code:", resp.content)

#%%
import telebot
import requests
from io import BytesIO
#%%
# Save the token in a variable
api_key = token
#declare the bot
bot = telebot.TeleBot(api_key)
# Replace 'Owner telegram ID' with the Variable
Owner_Id = '5924158528'
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#%%
import requests
import cv2
import numpy as np
from keras import models

# Load pre-trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = models.load_model("model.weights.best.keras")
untuned = models.load_model("3x3_model.weights.best.keras")

bot.message_handler(commands=['start', 'help', 'send'])
def send_welcome(message):
    #here u can put ur function with comparing the text
    if (message.text in ("/help")):
        bot.reply_to(message,"Drop an image.")
    elif (message.text in ("/send")):
        bot.reply_to(message,"Invalid input. Drop an image again.")
    else :
        bot.reply_to(message, "Welcome to Emotion detection system.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Lấy file_id của ảnh
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)

    # Tải ảnh xuống
    file_url = f'https://api.telegram.org/file/bot{token}/{file_info.file_path}'
    response = requests.get(file_url)

    # Đọc ảnh vào OpenCV
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Nhận diện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    # Nếu có ít nhất một khuôn mặt được nhận diện
    if len(faces) > 0:
        # Lấy tọa độ của khuôn mặt đầu tiên
        x, y, w, h = faces[0]
        # Cắt ảnh khuôn mặt từ frame
        face_roi = gray_frame[y:y+h, x:x+w]
        # Resize ảnh khuôn mặt về kích thước 48x48
        resized_face = cv2.resize(face_roi, (48, 48))
        # Mở rộng kích thước của ảnh để phù hợp với định dạng (1, 48, 48, 1)
        resized_face = resized_face.reshape((1, 48, 48, 1))

        # Chuẩn hóa ảnh
        resized_face = resized_face.astype('float32')
        resized_face /= 255

        # Dự đoán bằng mô hình
        prediction = model.predict(resized_face)
        predicted_label = label_names[np.argmax(prediction)]
        bot.reply_to(message,predicted_label)
    else :
        bot.reply_to(message,"Can't detect any people in image.")
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Don't say anything pls. Drop an image instead!!!")
#%%
bot.polling(none_stop=True)