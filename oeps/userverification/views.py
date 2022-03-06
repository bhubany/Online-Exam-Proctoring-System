from concurrent.futures import thread
import imp
from importlib.resources import path
from io import BytesIO
import OpenSSL
import threading
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
import base64
import time
import cv2
from PIL import Image
from numpy import asarray
import numpy as np
from io import StringIO
import os
from os import listdir
from tkinter import Frame
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
from mtcnn.mtcnn import MTCNN
import pickle
import cv2
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy
import mysql.connector
import re
from .models import *
from django.contrib.auth import login, authenticate, logout
from django.views.decorators.csrf import csrf_exempt

MyFaceNet = load_model('static/models/facenet_keras.h5')
regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'  # FOR EMAIL VALIDATION
# folder='bhuban'


# Create your views here.

def index(request):
    return render(request,'userverification/index.html')

# registration form
def register(request):
    return render(request,'userverification/register.html')

# Getting values from form to register
def register_details(request):
    reg_conf_pwd=reg_pwd=reg_role=reg_phone=reg_username=reg_email=reg_l_name=reg_m_name=reg_f_name=''
    if request.method=="POST":
        reg_f_name=str(request.POST['r_first_name'])
        reg_m_name=str(request.POST['r_middle_name'])
        reg_l_name=str(request.POST['r_last_name'])
        reg_email=str(request.POST['r_email'])
        reg_username=str(request.POST['r_username'])
        reg_phone=str(request.POST['r_phone'])
        reg_role=str(request.POST['r_role'])
        reg_pwd=str(request.POST['r_pwd'])
        reg_conf_pwd=str(request.POST['r_conf_pwd'])

    # =======USER VALIDATIOn==========
    reg_err=False
    reg_f_name_err=''
    reg_m_name_err=''
    reg_l_name_err=''
    reg_email_err=''
    reg_username_err=''
    reg_phone_err=''
    reg_role_err=''
    reg_pwd_err=''    
    if (not(reg_f_name.isalpha()) or len(reg_f_name)>20 or len(reg_f_name)==0):
        reg_f_name_err="Name must me Character of maximum length 20"
        reg_err=True
    if (len(reg_m_name)>0):
        if (not(reg_m_name.isalpha()) or len(reg_m_name)>20):
            reg_m_name_err="Name must me Character of maximum length 20"
            reg_err=True
    if (not(reg_l_name.isalpha()) or len(reg_l_name)>20 or len(reg_l_name)==0):
        reg_l_name_err="Name must me Character of maximum length 20"
        reg_err=True
    if(not(re.search(regex,reg_email)) or len(reg_email)>30 or len(reg_email)==0):
        reg_email_err = "Invalid Email"
        reg_err=True 
    elif User_info.objects.filter(u_email=reg_email).exists():
        reg_email_err="Email Already Taken"
        reg_err=True
    if (not(reg_username.isalnum()) and len(reg_username)<=20):
        reg_username_err="Usernaem must me Character of maximum length 20"
        reg_err=True
    elif User_info.objects.filter(u_username=reg_username).exists():
        reg_username_err="Username Already Taken"
        reg_err=True
    if (not(reg_phone.isnumeric()) and len(reg_phone)!=10):
        reg_phone_err="Phone Number must me integer and of 10 digit"
        reg_err=True
    if (not(reg_role.isnumeric()) or len(reg_role)!=1):
        reg_role_err="Please Select your role"
        reg_err=True
    if((reg_pwd==reg_conf_pwd)):
        if len(reg_pwd)>=8 and len(reg_conf_pwd)<=20:
            r_pwd=reg_pwd
        else:
          reg_pwd_err="Password must be of minimum 8 and maximum 20 Characters" 
          reg_err=True
    else:
        reg_pwd_err="Both Password didnot matched"  
        reg_err=True
     
    if reg_err==False:
        mydb =User_info(u_f_name=reg_f_name, u_m_name=reg_m_name, u_l_name=reg_l_name, u_email=reg_email, u_phone=reg_phone, u_username=reg_username, u_pass=r_pwd)
        try:
            mydb.save()
            os.makedirs('static/trained_data/'+str(reg_username))
            request.session['u_username']=reg_username
            return render(request,'userverification/face_registration.html',{'success_msg':"You have been registered sucessfully. Try Registering your face"})
        except:
            return render(request,'userverification/register.html',{'failure_msg': "Error Occurs Try again later"})            
    else:
        r_err={
            'reg_f_name_err':reg_f_name_err,
            'reg_m_name_err':reg_m_name_err,
            'reg_l_name_err':reg_l_name_err,
            'reg_email_err': reg_email_err,
            'reg_username_err':reg_username_err,
            'reg_phone_err':reg_phone_err,
            'reg_role_err': reg_role_err,
            'reg_pwd_err':reg_pwd_err,
            'reg_f_name':reg_f_name,
            'reg_m_name':reg_m_name,
            'reg_l_name':reg_l_name,
            'reg_email':reg_email,
            'reg_username':reg_username,
            'reg_phone':reg_phone,
            'reg_role':reg_role,
            'reg_pwd':reg_pwd,
            'reg_conf_pwd':reg_conf_pwd,
        }
        # print(r_err)
        return render(request,'userverification/register.html',(r_err))
    

# =====Registering User Faces
def user_face_registration(request):
    img_name=''
    decoded_image_data=''
    
    if 'u_username' in request.session:
        user_username=request.session['u_username']
        try:
            if request.method=="POST":
                img = request.POST['user_image']
                reg_img_count= request.POST['img_count']
                if len(reg_img_count)==0:
                    reg_img_count = 1
                else:
                    reg_img_count= int(reg_img_count)+1
                
                retake_image=request.POST['retake_image']
                if retake_image:
                    reg_img_count=reg_img_count-2
                    img_name=str(user_username)+str((reg_img_count))                
                    os.remove('static/user_image/dataset/'+img_name+'.png')
                else:
                    img_name = str(user_username)+str(reg_img_count)
                print("Image Name ==> {}, Image Count ==>{}".format(img_name,reg_img_count))
                img = img.replace('data:image/png;base64,','')
                img = img.replace(' ','+')
                seconds=time.time()
                decoded_image_data = base64.b64decode(img)
            

            with open('static/trained_data/'+user_username+'/'+img_name+'.png', 'wb') as file_to_save:
                    file_to_save.write(decoded_image_data)
            

            img1 = Image.open('static/user_image/dataset/'+img_name+'.png')
            img1 = img1.convert('RGB')  
            pixels = asarray(img1)
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(pixels)
            single_face="nooooo"
            if len(faces)>=1:
                single_face=True
                for face in faces:
                    x, y, width, height = face['box'] #for Face
                    left_eyeX, left_eyeY =face['keypoints']['left_eye']
                    right_eyeX, right_eyeY =face['keypoints']['right_eye']
                    noseX, noseY =face['keypoints']['nose']
                    left_mouthX, left_mouthY =face['keypoints']['mouth_left']
                    right_mouthX, right_mouthY =face['keypoints']['mouth_right']

                    # Drawing Rectangle on image using above values
                    cv2.rectangle(pixels, pt1=(x,y), pt2=(x+width, y+height), color=(255,255,255), thickness=3)

                    # Drawing Facial LendMarks
                    cv2.circle(pixels, center=(left_eyeX,left_eyeY), color=(255,255,255), thickness=1,radius=5)
                    cv2.circle(pixels, center=(right_eyeX,right_eyeY), color=(255,255,255), thickness=1,radius=5)
                    cv2.circle(pixels, center=(noseX,noseY), color=(255,255,255), thickness=1,radius=5)
                    cv2.circle(pixels, center=(left_mouthX,left_mouthY), color=(255,255,255), thickness=1,radius=5)
                    cv2.circle(pixels, center=(right_mouthX,right_mouthY), color=(255,255,255), thickness=1,radius=5)
                img2=Image.fromarray(pixels,'RGB')
                # im = Image.open("test.jpg")
                data =BytesIO()
                img2.save(data, "JPEG")
                encoded_img_data = base64.b64encode(data.getvalue())
                img2.save("static/user_image/dataset/decoded_image.png")
                list = os.listdir(dir)
                number_files = len(list)
                print (number_files)
            else:
                single_face=False
                print("Cannot detect Face")
            
            print(single_face)
            
            return render(request, 'userverification/face_registration.html',({'img2':encoded_img_data.decode('utf-8'), 'reg_img_count':int(reg_img_count),'retake':True}))
            # render(request, "stu_profile.html", {'userObj': user_objects})
        except:
            return render(request,'userverification/face_registration.html',{'failure_msg': "Capture Image"})
    else:
        return render(request,'userverification/register.html',{'failure_msg': "Error Occurs Try Registering"})    

# Login Form
def login(request):
    if 'u_username' in request.session:
        username=request.session['u_username']
        user=User_info.objects.filter(u_username= username)
        if len(user)>0:
            for u in user:
                if u.is_img_registered==0:
                    return render(request, 'userverification/face_registration.html', ({'failure_msg':"Hello "+username+", You haven\'t registered your face yet. Please Registered it"}))
                else:            
                    return render(request,'userverification/user_profile.html')
    else:
        return render(request,'userverification/login.html')

def logout(request):
    try:
        del request.session['u_username']
        return render(request,'userverification/login.html')
    except:
        return render(request,'userverification/user_profile.html',({'failure_msg':"Error occurs try again later"}))

# Checking Login Credentials obtained from login form
def check_login(request):
    if request.method=="POST":
        username = request.POST['login_username']
        pwd = request.POST['login_password']

        user=User_info.objects.filter(u_username= username) & User_info.objects.filter(u_pass = pwd)
        if len(user)>0:
            request.session['u_username']=username
            for u in user:
                if u.is_img_registered==0:
                    return render(request, 'userverification/face_registration.html', ({'failure_msg':"Hello "+username+", You haven\'t registered your face yet. Please Registered it",'username':username}))
                else:            
                    return render(request, 'userverification/user_profile.html', ({'success_msg':"Welcome back "+username}))
        else:
            print(username+pwd)
            return render(request, 'userverification/login.html',({'failure_msg':"Invalid login credintals"}))





# ============= Performing Training =============
@csrf_exempt
def train_module(request):
    print("Getting Request")
    if request.method == "POST":
        # folderName=request.session['u_username']
        folderName=request.POST['u_username']
        folder = 'static/trained_data/'+folderName+"/"
        database = {}
        faces=list()
        for filename in listdir(folder):
            # print("=============>")
            # print(folder + filename)
            if filename=="decoded_image.png":
                pass
            else:
                userImage = cv2.imread(folder + filename)    
                pixels = asarray(userImage)
        # create the detector, using default weights
            detector = MTCNN()
        # detect faces in the image
            result = detector.detect_faces(pixels)
        if len(result)<=0:        
            x1, y1, width, height = 0, 0, 0, 0        
        else:        
            x1, y1, width, height = result[0]['box'] 
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
        
            # image = userImage.convert('RGB')
            image = Image.fromarray(userImage)
            image_array = asarray(image)
        
            cropped_face = image_array[y1:y2, x1:x2]     # Face After cropping with coordinates value                   
            face = Image.fromarray(cropped_face)                       
            face = face.resize((160,160))
            c_face_array = asarray(face)
            x = c_face_array.astype('float32')
            mean, std = x.mean(), x.std()
            face_value = (x - mean) / std
            e_face_value = expand_dims(face_value, axis=0)
            signature = MyFaceNet.predict(e_face_value)
            faces.extend(signature)
        i=1
        for face in faces:
            if i==1:
                temp=face
            else:
                temp =numpy.add(temp,face)
            i=i+1
        
        final = numpy.true_divide(temp,len(faces))
        signature=asarray(final)

        database[folderName]=signature
        try:
            myfile = open("static/pickleFile/"+folderName+".pkl", "wb")
            pickle.dump(database, myfile)
            user_obj = User_info.objects.get(u_username=folderName)
            user_obj.is_img_registered= 1
            user_obj.save()
            myfile.close()
            # return render(request, 'userverification/user_profile.html',({'success_msg':"You face has been registered sucessfully"}))
            return JsonResponse({'status':1,})
        except:
            # return render(request, 'userverification/face_registration.html',({'failure_msg':"Error occurs try again later"}))
            return JsonResponse({'status':0,})
        # finally:

# ===========TEST==========


def home(request):
    return render(request, 'userverification/openCamera.html')
    

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
            )

@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error Occurs")
        pass

# cam.release() #TO close Camera


# ================
def test(request):
    # value=User_info()
    user=User_info.objects.all()
    return render(request, 'userverification/test.html',({'user':user}))

# @csrf_exempt #dont use this method due to security
def savetest(request):
    if request.method == "POST":
        uid=request.POST['uid']
        password =request.POST['password']
        phone = request.POST['phone']
        username = request.POST['username']
        email = request.POST['email']
        lastName = request.POST['lname']
        middleName = request.POST['mname']
        firstName = request.POST['fname']

        if uid=='': #adding new Records
            mydata =User_info(u_f_name=firstName, u_m_name=middleName, u_l_name=lastName, u_email=email,
                        u_phone=phone, u_username=username, u_pass=password)
        else:
            mydata =User_info(u_id =uid, u_f_name=firstName, u_m_name=middleName, u_l_name=lastName, u_email=email,
                        u_phone=phone, u_username=username, u_pass=password)
        mydata.save()
        usr=User_info.objects.values()
        usr_data = list(usr)
        # print(usr_data)
        return JsonResponse({'status':'save', 'usr_data':usr_data})
    else:
        return JsonResponse({'status':0})

def deleteData(request):
    if request.method == "POST":
        id = request.POST['sid']
        print(id)
        pi= User_info.objects.get(u_id=id)
        pi.delete()
        return JsonResponse({'status':1})
    else:
        return JsonResponse({'status':0})

def editData(request):
    if request.method == "POST":
        id = request.POST.get('sid')
        print(id)
        pi= User_info.objects.get(u_id=id)
        usr_data={"id":pi.u_id, "fname":pi.u_f_name,"lname":pi.u_l_name}
        return JsonResponse(usr_data)
    else:
        return JsonResponse({'status':0})

def camTest(request):
    return render(request, 'userverification/camTest.html')

@csrf_exempt
def faceTest(request):
    if request.method == "POST":
        usr_img=request.POST.get('img_url')
        img_id=request.POST.get('img_id')
        print(usr_img)
        user_username="bhuban"
        
        img = usr_img.replace('data:image/png;base64,','')
        img = img.replace(' ','+')
        seconds=time.time()
        decoded_image_data = base64.b64decode(img)
        img_name="img1"+img_id
        

        with open('static/trained_data/'+user_username+'/'+img_name+'.png', 'wb') as file_to_save:
                file_to_save.write(decoded_image_data)
        

        img1 = Image.open('static/trained_data/'+user_username+'/'+img_name+'.png')
        img1 = img1.convert('RGB')  
        pixels = asarray(img1)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        single_face="nooooo"
        if len(faces)>=1:
            single_face=True
            for face in faces:
                x, y, width, height = face['box'] #for Face
                left_eyeX, left_eyeY =face['keypoints']['left_eye']
                right_eyeX, right_eyeY =face['keypoints']['right_eye']
                noseX, noseY =face['keypoints']['nose']
                left_mouthX, left_mouthY =face['keypoints']['mouth_left']
                right_mouthX, right_mouthY =face['keypoints']['mouth_right']

                # Drawing Rectangle on image using above values
                cv2.rectangle(pixels, pt1=(x,y), pt2=(x+width, y+height), color=(5, 231, 247), thickness=3)

                # Drawing Facial LendMarks
                cv2.circle(pixels, center=(left_eyeX,left_eyeY), color=(0,255,0), thickness=3,radius=5)
                cv2.circle(pixels, center=(right_eyeX,right_eyeY), color=(0,255,0), thickness=3,radius=5)
                cv2.circle(pixels, center=(noseX,noseY), color=(5, 231, 247), thickness=3,radius=5)
                cv2.circle(pixels, center=(left_mouthX,left_mouthY), color=(5, 231, 247), thickness=3,radius=5)
                cv2.circle(pixels, center=(right_mouthX,right_mouthY), color=(5, 231, 247), thickness=3,radius=5)
            img2=Image.fromarray(pixels,'RGB')
            # im = Image.open("test.jpg")
            data =BytesIO()
            img2.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            img2.save('static/trained_data/'+user_username+'/'+'decoded_image.png')
            img_path='static/trained_data/'+user_username+'/'+'decoded_image.png'
            # list = os.listdir(dir)
            # number_files = len(list)
            # print (number_files)
        else:
            single_face=False
            print("Cannot detect Face")
        # encoded_img_data=[]
        # encoded_img_data.extend(encoded_img_data)
        # print(encoded_img_data)
        return JsonResponse({'status':1,'faceDetectedImage':img_path})
    else:
        return JsonResponse({'status':0})
