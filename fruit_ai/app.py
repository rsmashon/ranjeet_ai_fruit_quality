import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import smtplib


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Fresh Apple','Fresh Avocado','Fresh Banana','Fresh Orange','Fresh Pomegranate','Rotten Apple','Rotten Avocado','Rotten Banana','Rotten Orange','Rotten Pomegranate']

try:
    @app.route("/")
    def index():
        return render_template("index.html")
except Exception as e:
    print(e)
    
user_email = input('please enter user"s email address : ')
    
try:
    @app.route("/upload", methods=["POST"])
    def upload():
        target = os.path.join(APP_ROOT, 'images/')
        print(target)
        if not os.path.isdir(target):
                os.mkdir(target)
        else:
            print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist("file"))
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
    
            new_model = load_model('models/fruit_quality_model.h5')
            new_model.summary()
            test_image = image.load_img('images\\'+filename,target_size=(64,64))
            test_image = image.img_to_array(test_image)
            test_image=test_image/255
            test_image = np.expand_dims(test_image, axis = 0)
            result = new_model.predict(test_image)
            print(result)
            result=np.argmax(result)
            print(result)
            if result == 0 :
                prediction = classes[0]
            elif result == 1:
                prediction = classes[1]
            elif result ==2:
                prediction = classes[2]
            elif result == 3:
                prediction = classes[3]
            elif result ==4:
                prediction = classes[4]
            elif result == 5:
                prediction = classes[5]
            elif result == 6:
                prediction = classes[6]
            elif result == 7:
                prediction = classes[7]   
            elif result == 8:
                prediction = classes[8]
            else:
                prediction = classes[9]
                
            server = smtplib.SMTP_SSL("smtp.gmail.com",465)
            server.login("anmoljain1013@gmail.com","madjrniqngujyxnl")
            server.sendmail("anmoljain1013@gmail.com", user_email , f"this quality of fruit is ---> {prediction}")
            print("sending mail")
            
        return render_template("template.html",image_name=filename, text=prediction)
    
except Exception as e :
    print(e)

try:
    @app.route('/upload/<filename>')
    def send_image(filename):
        return send_from_directory("images", filename)
except Exception as e:
    print(e)

if __name__ == "__main__":
    app.run(debug=False)
