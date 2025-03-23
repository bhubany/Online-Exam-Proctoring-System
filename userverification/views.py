from channels.generic.websocket import WebsocketConsumer
# from channels.consumer import SyncConsumer
# import multiprocessing as mp
import csv
from os.path import exists
from asgiref.sync import async_to_sync
# from concurrent.futures import thread
from ctypes.wintypes import RGB
from fileinput import filename
import hashlib
from importlib.resources import path
from io import BytesIO
# import OpenSSL
import threading
from cv2 import waitKey
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
import base64
import time
import cv2
from PIL import Image
# from flask import request
from numpy import asarray
import numpy as np
from io import StringIO
import os
import base64
from os import listdir
from tkinter import Frame
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
import numpy as np
# from rsa import PublicKey
from facenet_pytorch import MTCNN
# from mtcnn.mtcnn import MTCNN as OLDMTCNN
import torch
import pickle
import cv2
# from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy
import mysql.connector
import re
from .models import *
# from django.contrib.auth import login, authenticate, logout
from django.views.decorators.csrf import csrf_exempt
# import socketio
# import io
import json
from datetime import datetime
import math

# from multiprocessing import Process
from multiprocessing import Pool
import traceback

from keras.models import load_model
from keras_facenet import FaceNet

MyFaceNet = FaceNet()

regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'  # FOR EMAIL VALIDATION
# folder='bhuban'
font = cv2.FONT_HERSHEY_SIMPLEX

# Create your views here.


def sum(i):
    r = i*i
    return r


numbers = range(10000)


def my_processing(funName, valueList):
    if __name__ == '__main__':
        p = Pool()
        result = p.map(funName, valueList)
        p.close()
        p.join()
        return result

# def second_fun():
#     res = my_processing(sum, numbers)
#     if res==None:
#         pass
#     else:
#         print(res)
# second_fun()
# def functionalWork():
#     second_fun(request)
#     return JsonResponse({'status':1,'msg':"Error Occurs try again later",'message':"message"})

# Generating Exam details


def scheduleExam(request):
    if request.method == "POST":
        exam_err = False
        e_detail = str(request.POST['examDetails'])
        exam_facl = str(request.POST['examFacl'])
        result = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        print("Exam Facl==> {}".format(exam_facl))
        if exam_facl == 0:
            exam_err = True
        mydb = exam_details(
            e_code=result, e_details=e_detail, e_faculty=exam_facl)
        success = False
        try:
            mydb.save()
            success = True
        except:
            print("Error occurs while generating exam details")
        if (success == True) and (exam_err == False):
            exam = exam_details.objects.values()
            exam_data = list(exam)
            return JsonResponse({'status': '1', 'exam_data': exam_data})
        else:
            return JsonResponse({'status:0'})


# IS exam Started?
def checkExamStatus(request):
    exam_id = False
    exam_status = 0

    if 'u_username' in request.session:
        u_username = request.session['u_username']

    if request.method == "POST":
        exam_code = request.POST['examCode']
        u_username = request.POST['u_username']
        exam_key = str(u_username)+str(exam_code)
        # Check active status for exam
        if exam_details.objects.filter(e_code=exam_code, e_staus=1):
            exam_id = True
            if examinee_details.objects.filter(ex_username=u_username, ex_code=exam_code, ex_staus=1).exists():
                exam_status = 2
                print("Already Entered in Exam")
            else:
                mydb = examinee_details(
                    ex_username=u_username, ex_code=exam_code, ex_staus=1)
                try:
                    mydb.save()
                    exam_status = 1
                except:
                    exam_status = 0

        if exam_status == 1 and exam_id == True:
            return JsonResponse({'status': 1, 'msg': "Exam started"})
        elif (exam_id == True) and (exam_status == 0):
            return JsonResponse({'status': 0, 'msg': "Error occurs try again later"})
        elif (exam_id == True) and (exam_status == 2):
            return JsonResponse({'status': 2, 'msg': "You can't enter twice in exam"})
        else:
            return JsonResponse({'status': 3, 'msg': "Invalid Exam ID"})


# main Page
def index(request):
    return render(request, 'userverification/index.html')

# registration form


def register(request):
    if 'u_username' in request.session:
        return user_login(request)
    else:
        return render(request, 'userverification/register.html')
# login form


def user_login(request):
    return user_login_page(request)

# admin Login Page


def admin_login(request):
    return admin_login_page(request)

# Getting values from form to register


def register_details(request):
    reg_conf_pwd = reg_pwd = reg_role = reg_phone = reg_username = reg_email = reg_l_name = reg_m_name = reg_f_name = reg_facl = ''
    if request.method == "POST":
        reg_f_name = str(request.POST['r_first_name'])
        reg_m_name = str(request.POST['r_middle_name'])
        reg_l_name = str(request.POST['r_last_name'])
        reg_email = str(request.POST['r_email'])
        reg_username = str(request.POST['r_username'])
        reg_phone = str(request.POST['r_phone'])
        reg_role = str(request.POST['r_role'])
        reg_pwd = str(request.POST['r_pwd'])
        reg_conf_pwd = str(request.POST['r_conf_pwd'])
        reg_facl = str(request.POST['r_facl'])

    # =======USER VALIDATIOn==========
    reg_err = False
    reg_f_name_err = ''
    reg_m_name_err = ''
    reg_l_name_err = ''
    reg_email_err = ''
    reg_username_err = ''
    reg_phone_err = ''
    reg_role_err = ''
    reg_pwd_err = ''
    reg_facul_err = ''
    if (not (reg_f_name.isalpha()) or len(reg_f_name) > 20 or len(reg_f_name) == 0):
        reg_f_name_err = "Name must me Character of maximum length 20"
        reg_err = True
    if (len(reg_m_name) > 0):
        if (not (reg_m_name.isalpha()) or len(reg_m_name) > 20):
            reg_m_name_err = "Name must me Character of maximum length 20"
            reg_err = True
    if (not (reg_l_name.isalpha()) or len(reg_l_name) > 20 or len(reg_l_name) == 0):
        reg_l_name_err = "Name must me Character of maximum length 20"
        reg_err = True
    if (not (re.search(regex, reg_email)) or len(reg_email) > 30 or len(reg_email) == 0):
        reg_email_err = "Invalid Email"
        reg_err = True
    elif User_info.objects.filter(u_email=reg_email).exists():
        reg_email_err = "Email Already Taken"
        reg_err = True
    if (not (reg_username.isalnum()) and len(reg_username) <= 20):
        reg_username_err = "Usernaem must me Character of maximum length 20"
        reg_err = True
    elif User_info.objects.filter(u_username=reg_username).exists():
        reg_username_err = "Username Already Taken"
        reg_err = True
    if (not (reg_phone.isnumeric()) and len(reg_phone) != 10):
        reg_phone_err = "Phone Number must me integer and of 10 digit"
        reg_err = True
    if (not (reg_role.isnumeric()) or len(reg_role) != 1):
        reg_role_err = "Please Select your role"
        reg_err = True
    if (not (reg_facl.isalpha()) or len(reg_facl) == ''):
        reg_facul_err = "Please Select your Faculty"
        reg_err = True

    if ((reg_pwd == reg_conf_pwd)):
        if len(reg_pwd) >= 8 and len(reg_conf_pwd) <= 20:
            r_pwd = reg_pwd
        else:
            reg_pwd_err = "Password must be of minimum 8 and maximum 20 Characters"
            reg_err = True
    else:
        reg_pwd_err = "Both Password didnot matched"
        reg_err = True
    print("Registration ROle==>{}".format(reg_role))
    if reg_err == False:
        mydb = User_info(u_f_name=reg_f_name, u_m_name=reg_m_name, u_l_name=reg_l_name, u_email=reg_email,
                         u_phone=reg_phone, u_username=reg_username, u_pass=r_pwd, u_role=reg_role, u_facl=reg_facl)
        try:
            mydb.save()
            os.makedirs('static/trained_data/'+str(reg_username))
            request.session['u_username'] = reg_username
            return register_face(request)
        except Exception as e:
            print(f"Error occured: {e}")
            return render(request, 'userverification/register.html', {'failure_msg': "Error Occurs Try again later"})
    else:
        r_err = {
            'reg_f_name_err': reg_f_name_err,
            'reg_m_name_err': reg_m_name_err,
            'reg_l_name_err': reg_l_name_err,
            'reg_email_err': reg_email_err,
            'reg_username_err': reg_username_err,
            'reg_phone_err': reg_phone_err,
            'reg_role_err': reg_role_err,
            'reg_pwd_err': reg_pwd_err,
            'reg_f_name': reg_f_name,
            'reg_m_name': reg_m_name,
            'reg_l_name': reg_l_name,
            'reg_email': reg_email,
            'reg_username': reg_username,
            'reg_phone': reg_phone,
            'reg_role': reg_role,
            'reg_pwd': reg_pwd,
            'reg_conf_pwd': reg_conf_pwd,
            'reg_facul_err': reg_facul_err,
        }
        # print(r_err)
        return render(request, 'userverification/register.html', (r_err))

# directing to user face registration page


def register_face(request, msg={}):
    admin = False
    if request.session.get('admin_username') or request.session.get('u_username'):
        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            role = 'admin'
        else:
            user_username = request.session['u_username']
            role = 'student'

        # msg['success_msg']="You have been registered sucessfully!"
        msg['user_username'] = user_username
        return render(request, 'userverification/face_registration.html', msg)
    else:
        msg = {'failure_msg': "Please Provide your login credintals"}
        return login_page(request, msg)
        # return render(request,'userverification/login.html',())

# =====Registering User Faces


def user_face_registration(request):
    img_name = ''
    decoded_image_data = ''

    if 'u_username' in request.session:
        user_username = request.session['u_username']
        try:
            if request.method == "POST":
                img = request.POST['user_image']
                reg_img_count = request.POST['img_count']
                if len(reg_img_count) == 0:
                    reg_img_count = 1
                else:
                    reg_img_count = int(reg_img_count)+1

                retake_image = request.POST['retake_image']
                if retake_image:
                    reg_img_count = reg_img_count-2
                    img_name = str(user_username)+str((reg_img_count))
                    os.remove('static/user_image/dataset/'+img_name+'.png')
                else:
                    img_name = str(user_username)+str(reg_img_count)
                print("Image Name ==> {}, Image Count ==>{}".format(
                    img_name, reg_img_count))
                img = img.replace('data:image/png;base64,', '')
                img = img.replace(' ', '+')
                seconds = time.time()
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
            single_face = "nooooo"
            if len(faces) >= 1:
                single_face = True
                for face in faces:
                    x, y, width, height = face['box']  # for Face
                    left_eyeX, left_eyeY = face['keypoints']['left_eye']
                    right_eyeX, right_eyeY = face['keypoints']['right_eye']
                    noseX, noseY = face['keypoints']['nose']
                    left_mouthX, left_mouthY = face['keypoints']['mouth_left']
                    right_mouthX, right_mouthY = face['keypoints']['mouth_right']

                    # Drawing Rectangle on image using above values
                    cv2.rectangle(pixels, pt1=(x, y), pt2=(
                        x+width, y+height), color=(255, 255, 255), thickness=3)

                    # Drawing Facial LendMarks
                    cv2.circle(pixels, center=(left_eyeX, left_eyeY),
                               color=(255, 255, 255), thickness=1, radius=5)
                    cv2.circle(pixels, center=(right_eyeX, right_eyeY),
                               color=(255, 255, 255), thickness=1, radius=5)
                    cv2.circle(pixels, center=(noseX, noseY), color=(
                        255, 255, 255), thickness=1, radius=5)
                    cv2.circle(pixels, center=(left_mouthX, left_mouthY), color=(
                        255, 255, 255), thickness=1, radius=5)
                    cv2.circle(pixels, center=(right_mouthX, right_mouthY), color=(
                        255, 255, 255), thickness=1, radius=5)
                img2 = Image.fromarray(pixels, 'RGB')
                # im = Image.open("test.jpg")
                data = BytesIO()
                img2.save(data, "JPEG")
                encoded_img_data = base64.b64encode(data.getvalue())
                img2.save("static/user_image/dataset/decoded_image.png")
                list = os.listdir(dir)
                number_files = len(list)
            else:
                single_face = False
                print("Cannot detect Face")

            print(single_face)

            return render(request, 'userverification/face_registration.html', ({'img2': encoded_img_data.decode('utf-8'), 'reg_img_count': int(reg_img_count), 'retake': True}))
            # render(request, "stu_profile.html", {'userObj': user_objects})
        except:
            return render(request, 'userverification/face_registration.html', {'failure_msg': "Capture Image"})
    else:
        return render(request, 'userverification/register.html', {'failure_msg': "Error Occurs Try Registering"})

# Login Form


def user_login_page(request, msg={}):
    if request.session.get('u_username'):
        user_username = request.session['u_username']
        print("Current User Username===>{}".format(user_username))

        if request.session.get(user_username):
            msg['success_msg'] = "Welcome back "+user_username
            msg['user_username'] = user_username
            return user_profile(request, msg)  # redirect to admin Profile
        else:
            msg['success_msg'] = "Welcome back " + \
                user_username+", Please verify your face"
            msg['admin_username'] = user_username
            return face_verification(request, msg)
    else:
        return render(request, 'userverification/userLogin.html', msg)

# Rendering Admin Login page


def admin_login_page(request, msg={}):
    if request.session.get('admin_username'):
        admin_username = request.session['admin_username']
        if request.session.get(admin_username):
            msg['success_msg'] = "Welcome back "+admin_username
            msg['admin_username'] = admin_username
            return admin_profile(request, msg)  # redirect to admin Profile
        else:
            msg['success_msg'] = "Welcome back " + \
                admin_username+", Please verify your face"
            msg['user_username'] = admin_username
            return face_verification(request, msg)
    else:
        return render(request, 'userverification/admin_login.html')

# validating admin login details


def check_admin_login(request, msg={}):
    if request.session.get('admin_username'):
        return admin_login_page(request)
    else:
        if request.method == "POST":
            admin_username = request.POST['admin_login_username']
            admin_pwd = request.POST['admin_login_password']
            user = User_info.objects.filter(u_username=admin_username) & User_info.objects.filter(
                u_pass=admin_pwd) & User_info.objects.filter(u_role=2)
            if len(user) > 0:
                request.session['admin_username'] = admin_username
                msg['success_msg'] = "Welcome back " + \
                    admin_username+", Please verify your face"
                msg['user_username'] = admin_username
                return face_verification(request, msg)
            else:
                msg['failure_msg'] = "Invalid login Credintals"
                return admin_login(request)
        else:
            msg['failure_msg'] = "Invalid login Credintals"
            return admin_login(request, msg)


def logout(request):
    if request.session.get('u_username'):

        user_username = request.session['u_username']
        if request.session.get(user_username):
            del request.session[user_username]
            del request.session['u_username']
            return user_login_page(request, msg={"failure_msg": "Try login"})
        else:
            del request.session['u_username']
            return user_login_page(request, msg={"failure_msg": "Try login"})
    elif request.session.get('admin_username'):
        user_username = request.session['admin_username']

        if request.session.get(user_username):
            del request.session[user_username]
            del request.session['admin_username']
            return admin_login_page(request, msg={"failure_msg": "Try login"})
        else:
            del request.session['admin_username']
            return admin_login_page(request, msg={"failure_msg": "Try login"})
    else:
        return index(request)


# Checking Login Credentials obtained from login form
def check_user_login(request, msg={}):
    if request.session.get('u_username'):
        return user_login_page(request)
    else:
        if request.method == "POST":
            user_username = request.POST['login_username']
            pwd = request.POST['login_password']

            user = User_info.objects.filter(u_username=user_username) & User_info.objects.filter(
                u_pass=pwd) & User_info.objects.filter(u_role=1)
            if len(user) > 0:
                request.session['u_username'] = user_username
                msg['success_msg'] = "Welcome back " + \
                    user_username+", Please verify your face"
                msg['user_username'] = user_username
                return face_verification(request, msg)
            else:
                msg['failure_msg'] = "Invalid login Credintals"
                return user_login_page(request, msg)

        else:
            msg['failure_msg'] = "Invalid login Credintals"
            return user_login_page(request, msg)


def face_verification(request, msg={}):
    if request.session.get('admin_username') or request.session.get('u_username'):

        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            user_role = 'admin'
        else:
            user_role = 'student'
            user_username = request.session['u_username']

        user = User_info.objects.filter(u_username=user_username)
        if len(user) > 0:
            print("User==>{}".format(user[0]))
            for u in user:
                if u.is_img_registered == 0:
                    msg['failure_msg'] = "Hello "+user_username + \
                        ", Please register your face"
                    msg['user_username'] = user_username
                    return register_face(request, msg)
                else:
                    msg['success_msg'] = "Hello "+user_username + \
                        ", Welcome back \n Please Verify It's You :)"
                    msg['user_username'] = user_username
                    return render(request, 'userverification/user_face_verification.html', msg)
        else:
            return logout(request)
    else:
        msg = {'failure_msg': "Please provide your login credintals"}
        return user_login_page(request, msg)


def admin_profile(request, msg={}):
    if request.session.get('admin_username'):
        admin_username = request.session['admin_username']
        if request.session.get(admin_username):
            msg['success_msg'] = "Welcome Back "+admin_username
            msg['admin_username'] = admin_username
            exam = exam_details.objects.values()
            exam_data = list(exam)
            msg['exam_data'] = exam_data
            print(msg)
            return render(request, 'userverification/admin_profile.html', msg)
        else:
            msg['success_msg'] = "Welcome Back "+admin_username
            msg['admin_username'] = admin_username
            return face_verification(request, msg)
    else:
        msg['failure_msg'] = "Please provide your login credintals"
        return admin_login(request, msg)


# ============= Performing Training =============

def train_module(request):
    temp = ''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if request.session.get('admin_username') or request.session.get('u_username'):

        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            user_role = 'admin'
        else:
            user_role = 'student'
            user_username = request.session['u_username']

        if request.method == "POST":
            folderName = str(user_username)
            folder = 'static/trained_data/'+folderName+"/"
            database = {}
            pixels = ''
            faces = list()

            for filename in listdir(folder):
                if filename != "decoded_image.png" and filename != "face_verification.png":
                    userImage = cv2.imread(folder + filename)
                    pixels = asarray(userImage)
                    mtcnn = MTCNN(keep_all=True,
                                  image_size=160, margin=30, min_face_size=50,
                                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                                  device=device)
                    result, conf, = mtcnn.detect(pixels, landmarks=False)
                    if (result is None):
                        x1, y1, width, height = 0, 0, 0, 0
                        print("Cant detect Face")
                    else:
                        print("Face Detected")
                        x1, y1, width, height = result[0]
                        x1, y1 = int(x1), int(y1)
                        x2, y2 = int(width), int(height)

                        # image = userImage.convert('RGB')
                        image = Image.fromarray(userImage)
                        image_array = asarray(image)

                        # Face After cropping with coordinates value
                        cropped_face = image_array[y1:y2, x1:x2]
                        face = Image.fromarray(cropped_face)
                        face = face.resize((160, 160))
                        c_face_array = asarray(face)
                        x = c_face_array.astype('float32')
                        mean, std = x.mean(), x.std()
                        face_value = (x - mean) / std
                        e_face_value = expand_dims(face_value, axis=0)
                        signature = MyFaceNet.embeddings(e_face_value)
                        faces.extend(signature)
            i = 1
            for face in faces:
                if i == 1:
                    temp = face
                else:
                    temp = numpy.add(temp, face)
                i = i+1

            final = numpy.true_divide(temp, len(faces))
            signature = asarray(final)

            database[folderName] = signature
            try:
                myfile = open("static/pickleFile/"+folderName+".pkl", "wb")
                pickle.dump(database, myfile)
                user_obj = User_info.objects.get(u_username=folderName)
                user_obj.is_img_registered = 1
                user_obj.save()
                myfile.close()
                print("Successfully Trained")

                return JsonResponse({'status': 1})
            except:

                return JsonResponse({'status': 0, 'msg': "Error occurs try again later"})
            # finally:
    else:
        return user_login_page(request, msg={'failure_msg': "Try login first"})

# when training is success


def face_training_success(request, msg={}):
    if request.session.get('u_username') or request.session.get('admin_username'):
        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            user_role = 'admin'
        else:
            user_role = 'student'
            user_username = request.session['u_username']

        msg["success_msg"] = "hello "+user_username + \
            ", your face hasbeen registered successfully. Please verify it for further process:)"
        msg["u_username"] = user_username

        return face_verification(request, msg)
    else:
        return user_login_page(request, msg={'failure_msg': "Try login first"})

# Face Recognition


def face_recognition(fileName, frame):
    facenetModel = MyFaceNet
    myfile = open("static/pickleFile/"+fileName+".pkl", "rb")

    database = pickle.load(myfile)
    myfile.close()
    accuracy = ' '
    identity = ''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # detect faces in the image
    mtcnn = MTCNN(keep_all=True, image_size=160, margin=30, min_face_size=50,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                  device=device)

    pixels = asarray(frame)

    bounding_boxes, conf, landmarks = mtcnn.detect(pixels, landmarks=True)

    if (bounding_boxes is None):
        x1, y1, x2, y2 = 1, 1, 1, 1
        identity = "No Face"
        dist = 0
    else:
        for i in range(len(bounding_boxes)):
            x1, y1, x2, y2 = bounding_boxes[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), 2)
        for i in range(len(landmarks)):
            for p in range(landmarks[i].shape[0]):
                cv2.circle(frame,
                           (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])),
                           2, (0, 0, 255), -1, cv2.LINE_AA)
        x1, y1, width, height = bounding_boxes[0]
        x1, y1 = int(abs(x1)), int(abs(y1))
        x2, y2 = int(width), int(height)
        face = pixels[y1:y2, x1:x2]

        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std  # standard Normal

        face = expand_dims(face, axis=0)
        signature = facenetModel.embeddings(face)

        min_dist = 10

        for key, value in database.items():
            print("Matching Value of Signature")
            # TO calculate Euclidean Distance
            dist = distance.euclidean(value, signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
            else:
                identity = "Unknown Person"
                dist = 0
    cv2.putText(frame, identity, (7, 190), font,
                3, (100, 255, 0), 3, cv2.LINE_AA)
    print("identity==>{}".format(identity))
    return frame, identity, dist


# ===========Verifying User==========
def verify_user_face(request, msg={}):
    admin = False
    user_role = ''

    if request.session.get('admin_username') or request.session.get('u_username'):

        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            user_role = 'admin'
        else:
            user_role = 'student'
            user_username = request.session['u_username']

        if request.session.get(user_username):
            if admin == True:
                msg['success_msg'] = "Welcome Back "+user_username+" ! "
                msg['admin_username'] = user_username
                return admin_profile(request, msg)
            else:
                msg['success_msg'] = "Welcome Back "+user_username+" ! "
                msg['u_username'] = user_username
                return user_profile(request, msg)
        else:
            try:
                if request.method == "POST":
                    user_username = str(request.POST['username'])
                    usr_img = request.POST.get('img_url')
                    img = usr_img.replace('data:image/png;base64,', '')
                    img = img.replace(' ', '+')
                    # decoded_image_data = base64.b64decode(img)
                    im_bytes = base64.b64decode(img)
                    # im_arr is one-dim Numpy array
                    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
                    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

                    # print("Up to saving Image")
                    x1, y1, x2, y2, identity, dist, frame = face_recognitions(
                        fileName=user_username, facenetModel=MyFaceNet, frame=img)
                    cv2.putText(frame, identity, (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 2)
                    # frame, identity, dist=face_recognition(fileName=user_username,facenetModel=MyFaceNet,frame=img) #return image by drawing landmarks and boxes
                    # img2=Image.fromarray(frame,'RGB')
                    cv2.imwrite('static/trained_data/'+user_username +
                                '/'+'face_verification.png', frame)
                    img_path = 'static/trained_data/'+user_username+'/'+'face_verification.png'

                    if (identity == "No Face"):
                        return JsonResponse({'status': 0, 'msg': "NO Face Detected", 'img_path': img_path})
                    elif (identity == "Unknown Person"):
                        return JsonResponse({'status': 2, 'msg': "Unknown Person", 'img_path': img_path})
                    else:
                        request.session[user_username] = "faceVerified"
                        return JsonResponse({'status': 1, 'user': user_role, 'msg': "Face matched Successfully", 'img_path': img_path})
                else:
                    return JsonResponse({'status': 0, 'msg': "Error while submitting Image"})
            except Exception as e:
                print(f"Error occcured {e}")
                traceback.print_exc()
                return JsonResponse({'status': 0, 'msg': "Error Occurs try again later"})
    else:
        return user_login_page(request, msg={'failure_msg': "Try Login First"})

# Redirecting to user Profile after Successfull Verification


def user_profile(request, msg={}):
    faculty = ''
    if request.session.get('u_username'):
        user_username = request.session['u_username']
        if request.session.get(user_username):
            # to display their exams
            user = User_info.objects.filter(u_username=user_username)
            if len(user) > 0:
                for u in user:
                    faculty = u.u_facl
                    print("User_Faculty==>{}".format(faculty))
            e_details = exam_details.objects.all().filter(e_faculty=faculty)

            msg['success_msg'] = "Welcome Back "+user_username
            msg['user_username'] = user_username
            msg['e_details'] = e_details

            return render(request, 'userverification/user_profile.html', msg)
        else:
            msg['success_msg'] = "Welcome Back "+user_username
            msg['user_username'] = user_username
            return face_verification(request, msg)
    else:
        msg['failure_msg'] = "Please provide your login credintals"
        return user_login_page(request, msg)


# Giving Final Exam
def final_exam_page(request, msg={}):
    if 'u_username' in request.session:
        user_username = request.session['u_username']

        if user_username in request.session:
            val = request.session[user_username]
            msg['success_msg'] = "Welcome Back "+user_username+". "
            msg['user_username'] = user_username
            # exam_key = str(user_username)+"_exam_code"
            # if exam_key in request.session:
            #     msg['exam_code']=request.session[exam_key]

            return render(request, 'userverification/final_exam_page.html', msg)
        else:
            return face_training_success(request)
    else:
        return user_login_page(request, msg={'failure_msg': "Try Login First"})

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
        yield (b'--frame\r\n'
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
    user = User_info.objects.all()
    return render(request, 'userverification/test.html', ({'user': user}))

# @csrf_exempt #dont use this method due to security


def savetest(request):
    if request.method == "POST":
        uid = request.POST['uid']
        password = request.POST['password']
        phone = request.POST['phone']
        username = request.POST['username']
        email = request.POST['email']
        lastName = request.POST['lname']
        middleName = request.POST['mname']
        firstName = request.POST['fname']

        if uid == '':  # adding new Records
            mydata = User_info(u_f_name=firstName, u_m_name=middleName, u_l_name=lastName, u_email=email,
                               u_phone=phone, u_username=username, u_pass=password)
        else:
            mydata = User_info(u_id=uid, u_f_name=firstName, u_m_name=middleName, u_l_name=lastName, u_email=email,
                               u_phone=phone, u_username=username, u_pass=password)
        mydata.save()
        usr = User_info.objects.values()
        usr_data = list(usr)
        return JsonResponse({'status': 'save', 'usr_data': usr_data})
    else:
        return JsonResponse({'status': 0})


def deleteData(request):
    if request.method == "POST":
        id = request.POST['sid']
        pi = User_info.objects.get(u_id=id)
        pi.delete()
        return JsonResponse({'status': 1})
    else:
        return JsonResponse({'status': 0})


def editData(request):
    if request.method == "POST":
        id = request.POST.get('sid')
        pi = User_info.objects.get(u_id=id)
        usr_data = {"id": pi.u_id, "fname": pi.u_f_name, "lname": pi.u_l_name}
        return JsonResponse(usr_data)
    else:
        return JsonResponse({'status': 0})


def camTest(request):
    return render(request, 'userverification/camTest.html')

# detect faces and drawing landmarks


def detect_and_draw_on_face(frame):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, image_size=160, margin=30, min_face_size=50,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                  device=device)
    frame = np.asarray(frame)
    bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)
    count = 0
    if (bounding_boxes is None):
        cv2.putText(frame, "No Face", (7, 70), font,
                    3, (100, 255, 0), 3, cv2.LINE_AA)
        return frame, count
    else:
        if len(bounding_boxes) == 1:
            count = 1
            x1, y1, x2, y2 = bounding_boxes[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color=(255, 255, 255), thickness=3)

            for i in range(len(landmarks)):
                for p in range(landmarks[i].shape[0]):
                    cv2.circle(frame, center=(int(landmarks[i][p, 0]), int(
                        landmarks[i][p, 1])), color=(255, 255, 255), thickness=1, radius=5)
            return frame, count
        elif (len(bounding_boxes) > 1):
            count = 2
            for i in range(len(bounding_boxes)):
                x1, y1, x2, y2 = bounding_boxes[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 0, 255), 2)
            for i in range(len(landmarks)):
                for p in range(landmarks[i].shape[0]):
                    cv2.circle(frame,
                               (int(landmarks[i][p, 0]),
                                int(landmarks[i][p, 1])),
                               2, (0, 0, 255), 3, cv2.LINE_AA)
            return frame, count
        else:
            return frame, count

# Testing does face have clear image or not


@csrf_exempt
def face_registration(request, msg={}):
    admin = False
    user_role = ''

    if request.session.get('admin_username') or request.session.get('u_username'):

        if request.session.get('admin_username'):
            user_username = request.session['admin_username']
            admin = True
            user_role = 'admin'
        else:
            user_role = 'student'
            user_username = request.session['u_username']

        try:
            if request.method == "POST":
                usr_img = request.POST.get('img_url')
                img_id = request.POST.get('img_id')

                img = usr_img.replace('data:image/png;base64,', '')
                img = img.replace(' ', '+')
                # seconds=time.time()
                decoded_image_data = base64.b64decode(img)
                img_name = "img1"+img_id

                with open('static/trained_data/'+user_username+'/'+img_name+'.png', 'wb') as file_to_save:
                    file_to_save.write(decoded_image_data)

                img1 = Image.open('static/trained_data/' +
                                  user_username+'/'+img_name+'.png')
                img1 = img1.convert('RGB')
                # return image by drawing landmarks and boxes
                frame, count = detect_and_draw_on_face(img1)
                img2 = Image.fromarray(frame, 'RGB')
                # im = Image.open("test.jpg")
                data = BytesIO()
                img2.save(data, "JPEG")
                encoded_img_data = base64.b64encode(data.getvalue())
                img2.save('static/trained_data/' +
                          user_username+'/'+'decoded_image.png')
                img_path = 'static/trained_data/'+user_username+'/'+'decoded_image.png'

                return JsonResponse({'status': 1, 'faceDetectedImage': img_path})
            else:
                return JsonResponse({'status': 0})
        except:
            return JsonResponse({'status': 0, 'msg': "failure occurs try again later"})
    else:
        return index(request)

# Test


def test01(request):
    return render(request, 'userverification/test01.html')

# View Exam


def viewExam(request):
    if request.method == "POST":
        exam_id = request.POST['eId']
        faculty = request.POST['faculty']
        exam = exam_details.objects.filter(e_id=exam_id).filter(e_staus=1)
        e_code = ''
        if len(exam) > 0:
            return JsonResponse({'status': 1, 'exam_code': e_code, 'msg': "Exam Code matched Successfully"})
        else:
            return JsonResponse({'status': 0, 'msg': "Error Occurs try again later", 'exam_code': e_code})


def startExam(request):
    if request.method == "POST":
        exam_id = request.POST['eId']
        faculty = request.POST['faculty']
        exam = exam_details.objects.filter(e_id=exam_id)
        e_code = ''
        if len(exam) > 0:
            exam.update(e_staus=1)
            for e in exam:
                e_code = e.e_code
                request.session['exam_code'] = faculty
            return JsonResponse({'status': 1, 'exam_code': e_code, 'msg': "Exam Code matched Successfully"})
        else:
            return JsonResponse({'status': 0, 'msg': "Error Occurs try again later", 'exam_code': e_code})


def endExam(request):
    if request.method == "POST":
        exam_id = request.POST['eId']
        faculty = request.POST['faculty']
        exam = exam_details.objects.filter(e_faculty=faculty)
        e_code = ''
        if len(exam) > 0:
            exam.update(e_staus=0)
            return JsonResponse({'status': 1, 'exam_code': e_code, 'msg': "Exam Code matched Successfully and has been ended"})
        else:
            return JsonResponse({'status': 0, 'msg': "Error Occurs try again later", 'exam_code': e_code})


def examineeDetails(request):
    if request.session.get('admin_username') and request.session.get('exam_code'):
        exam_code = request.session.get('exam_code')

        examinee = User_info.objects.all().filter(u_facl=exam_code)
        # examinee.objects.get(ex_code=examCode)
        examinee_data = list(examinee)

        msg = {
            'exam_code': exam_code,
            'examinee_data': examinee_data,
        }

        return render(request, 'userverification/examineeDetails.html', msg)
    else:
        return admin_login(request)

# ===============SOcket.io and Django===


count = 0
tempImg = ''
newFrame = ''
prev_frame_time = 0
new_frame_time = 0


# import


class WSAdmin(WebsocketConsumer):
    # def connect(self):
    #     self.accept()

    #     # self.send(text_data = json.dumps({
    #     #     'text': 'Connection_established admin',
    #     #     'message': "You are now connected as admin"

    #     #     # image from saved storage with their unique username
    #     # }))

    #     t1 = threading.Thread(target=send_data)
    #     t1.start()

    # def receive(self, text_data):
    #     text_data_json = json.loads(text_data)

    # val="value"

    def send_data(self, data):
        for filename in listdir('static/exam_data/examsarba1234/'):
            img = cv2.imread('static/exam_data/examsarba1234/'+filename)

            frame = detect_multiple_person(img)
            # p1 = mp.Process(target=detect_multiple_person, args=(img, count,))
            # p1.start()
            # count+=1
            # frame = newFrame
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # time when we finish processing for this frame
            # new_frame_time = time.time()
            # fps = 1/(new_frame_time-prev_frame_time)
            # prev_frame_time = new_frame_time
            # fps = str(fps)

            _, im_arr = cv2.imencode('.jpg', frame)
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)

            self.send(text_data=json.dumps({
                'type': 'image',
                'message': im_b64.decode('utf-8'),
            }))

    def connect(self):
        self.accept()

        # self.send(text_data = json.dumps({
        #     'text': 'Connection_established admin',
        #     'message': "You are now connected as admin"

        #     # image from saved storage with their unique username
        # }))

        t1 = threading.Thread(target=self.send_data)
        t1.start()


class WSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

        # for i in range(10): if user is admin (But how to know user is admin?)
        self.send(text_data=json.dumps({
            'text': 'Connection_established',
            'message': "You are now connected"

            # image from saved storage with their unique username
        }))

    def receive(self, text_data):
        global count, prev_frame_time, new_frame_time
        csv_value = {}
        frame = ''
        recg = ''
        text_data_json = json.loads(text_data)
        usr_img = text_data_json['message']
        usr_username = text_data_json['user_username']
        exam_code = text_data_json['examCode']
        count = text_data_json['count']

        exam = exam_details.objects.filter(
            e_faculty=exam_code).filter(e_staus=0)
        if len(exam) > 0:
            self.send(text_data=json.dumps({
                'examWarning': 'True',
                'message': "Exam has been ended by admin"

                # image from saved storage with their unique username
            }))
        # user_exam_image = str(exam_code)+usr_username
        img_name = str(count)+str(exam_code)+usr_username
        # create folder if not created
        if os.path.isdir('static/exam_data/'+usr_username):
            pass
        else:
            os.makedirs('static/exam_data/'+usr_username)

        img = usr_img.replace('data:image/png;base64,', '')
        img = img.replace(' ', '+')

        decoded_image_data = base64.b64decode(img)
        # Saving Image To directory
        with open('static/exam_data/'+usr_username+'/'+img_name+'.png', 'wb') as file_to_save:
            file_to_save.write(decoded_image_data)

        frame = cv2.imread('static/exam_data/' +
                           usr_username+'/'+img_name+'.png')

        # for multiple/count person
        img, faceCount = detect_multiple_person(frame)
        print("Face count ==> {}".format(faceCount))
        csv_value['count'] = count
        csv_value['mp'] = faceCount

        # Face Recognition
        x1, y1, x2, y2, identity, dist, frame = face_recognitions(
            MyFaceNet, usr_username, frame)
        # print(identity)
        # if identity=='No Face':
        #     recg=0
        # elif identity=='unknown Person':
        #     recg=2
        # else:
        #     recg=1
        csv_value['recg'] = identity

        # For Head pose estimation
        img, pitch, yaw, roll, estimated_pose = head_pose(frame)
        csv_value['roll'] = roll
        csv_value['yaw'] = yaw
        csv_value['pitch'] = pitch
        csv_value['estimated_pose'] = estimated_pose

        summary = str(faceCount)+"-Person, " + \
            identity+" pose->"+estimated_pose
        csv_value['summary'] = summary

        # FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        # print("FPS==========================>{}".format(fps))
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        print("FPS=====>{}".format(fps))

        csv_value['extra'] = fps

        csvFilename = 'static/csv_exam_data/'+usr_username+'/'+usr_username+'.csv'
        if os.path.isdir('static/csv_exam_data/'+usr_username):
            pass
        else:
            os.makedirs('static/csv_exam_data/'+usr_username)
        store_data_in_csv(csvFilename, csv_value)  # save file to path

# writing output details to csv file for each user


def store_data_in_csv(pathname, value={}):
    if not exists(pathname):
        with open(pathname, 'a', newline='') as f:
            fieldnames = ['count', 'recognition', 'multiple_person',
                          'roll', 'yaw', 'pitch', 'extra', 'summary', 'estimated_pose']
            headerWriter = csv.DictWriter(f, fieldnames=fieldnames)
            headerWriter.writeheader()
            headerWriter.writerow({'count': value['count'], 'recognition': value['recg'], 'multiple_person': value['mp'], 'roll': value['roll'],
                                  'yaw': value['yaw'], 'pitch': value['pitch'], 'extra': value['extra'], 'summary': value['summary'], 'estimated_pose': value['estimated_pose']})
    else:
        with open(pathname, 'a', newline='') as f:
            fieldnames = ['count', 'recognition', 'multiple_person',
                          'roll', 'yaw', 'pitch', 'extra', 'summary', 'estimated_pose']
            headerWriter = csv.DictWriter(f, fieldnames=fieldnames)
            headerWriter.writerow({'count': value['count'], 'recognition': value['recg'], 'multiple_person': value['mp'], 'roll': value['roll'],
                                  'yaw': value['yaw'], 'pitch': value['pitch'], 'extra': value['extra'], 'summary': value['summary'], 'estimated_pose': value['estimated_pose']})

# For head pose detection


def head_pose(frame):
    estimated_pose = ''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    size = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # pixels = asarray(frame)              #convert the image to numpy array
    # detector = MTCNN()                  #assign the MTCNN detector

    # detect faces in the image
    # res = detector.detect_faces(pixels)
    mtcnn = MTCNN(keep_all=True, device=device)
    # frame = frame.convert('RGB')
    pixels = np.asarray(frame)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)

    if (bounding_boxes is None):
        x1, y1, x2, y2 = 0, 0, 0, 0
        identity = "No Face"
        dist = 0
        lefteye = (0, 0)
        righteye = (0, 0)
        nose = (0, 0)
        leftmouth = (0, 0)
        rightmouth = (0, 0)
    else:
        # landmarks=landmarks
        # if len(f)!=0:
        lefteye = landmarks[0][0]
        righteye = landmarks[0][1]
        nose = landmarks[0][2]
        leftmouth = landmarks[0][3]
        rightmouth = landmarks[0][4]

    image_points = np.array([
                            (nose[0], nose[1]),     # Nose tip
                            (0, 0),   # Chin
                            # Left eye left corner
                            (lefteye[0], lefteye[1]),
                            # Right eye right corne
                            (righteye[0], righteye[1]),
                            # Left Mouth corner
                            (leftmouth[0], leftmouth[1]),
                            # Right mouth corner
                            (rightmouth[0], rightmouth[1])
                            ], dtype="double")

# # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            # Right eye right corne
                            (165.0, 170.0, -135.0),
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    rvec_matrix, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rvec_matrix)

    eulerAngles = angles

    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]

    x = abs(pitch*math.cos(roll))
    y = abs(pitch*math.sin(roll))
    z = abs(roll)
    z = 10
    if ((x > z or y > z)):
        estimated_pose = "away"
    else:
        estimated_pose = "screen"

    # for p in image_points:
    #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    # cv2.putText(frame, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame, pitch, yaw, roll, estimated_pose

# actual face recognition


def face_recognitions(facenetModel, fileName, frame):
    folder = 'static/pickleFile/'+fileName+'.pkl'
    myfile = open(folder, "rb")
    database = pickle.load(myfile)
    myfile.close()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)
    # frame = frame.convert('RGB')
    pixels = np.asarray(frame)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)

    if (bounding_boxes is None):
        x1, y1, x2, y2 = 1, 1, 1, 1
        identity = "No Face"
        dist = 0
    else:
        value = landmarks[0][3]
        cv2.circle(frame, (int(value[0]), int(value[1])),
                   2, (0, 0, 255), -1, cv2.LINE_AA)
        x1, y1, width, height = bounding_boxes[0]
        x1, y1 = int(abs(x1)), int(abs(y1))
        x2, y2 = int(width), int(height)
        face = pixels[y1:y2, x1:x2]

        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std

        face = expand_dims(face, axis=0)
        signature = facenetModel.embeddings(face)

        min_dist = 5
        identity = ''
        for key, value in database.items():
            # TO calculate Euclidean Distance
            dist = distance.euclidean(value, signature[0])
            if dist < min_dist:
                min_dist = dist
                identity = key
            else:
                identity = "unknown Person"
                dist = 0
    return x1, y1, x2, y2, identity, dist, frame

# multiple person detection


def detect_multiple_person(frame):
    faceCount = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # detector = MTCNN()
    # ====
    mtcnn = MTCNN(keep_all=True, device=device)
    # frame = frame.convert('RGB')
    frame = np.asarray(frame)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)

    if (bounding_boxes is None):
        return frame, faceCount
    else:
        for face in bounding_boxes:
            faceCount += 1
            x1, y1, x2, y2 = bounding_boxes[0]
            if (type(bounding_boxes) != "NoneType"):
                for i in range(len(bounding_boxes)):
                    x1, y1, x2, y2 = bounding_boxes[i]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 2)
                for i in range(len(landmarks)):
                    for p in range(landmarks[i].shape[0]):
                        cv2.circle(frame,
                                   (int(landmarks[i][p, 0]),
                                    int(landmarks[i][p, 1])),
                                   2, (0, 0, 255), -1, cv2.LINE_AA)
        return frame, faceCount


def getOutputValues(request):
    if request.method == "POST":
        examCode = request.POST.get('examCode')
        count = request.POST.get('count')
        # global_exam_count

        # select * connected user username
        examinee = User_info.objects.all().filter(u_facl=examCode)
        # examinee.objects.get(ex_code=examCode)
        examinee_data = list(examinee)
        usernames = []
        mainOutputList = []
        for ex in examinee_data:
            usernames.append(str(ex.u_username))

        for u_name in usernames:
            filename = 'static/csv_exam_data/'+u_name+'/'+u_name+'.csv'
            if not exists(filename):
                print("File doesnot exist")
                rows = list()
                mainOutputList.append(['0', '0', '0', '0', '0', '0', '0'])
                mainOutputList.append(u_name)
            else:
                with open(filename) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    rows = list(csv_reader)
                    mainOutputList.append(rows[int(count)])
                    mainOutputList.append(u_name)
        return JsonResponse({'status': 1, 'msg': "Operation Performed Successfully", 'mainOutputList': mainOutputList})
    else:
        return JsonResponse({'status': 0, 'msg': "Error Occurs try again later"})


def detect_multiple_persons(frame):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # detector = MTCNN()
    # ====
    mtcnn = MTCNN(keep_all=True, device=device)
    # frame = frame.convert('RGB')
    frame = np.asarray(frame)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)
    message = ''

    if (bounding_boxes is None):
        _, im_arr = cv2.imencode('.jpg', frame)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)

        # self.send(text_data = json.dumps({
        #     'type': 'image',
        message = im_b64.decode('utf-8')
        return message

    else:
        for _ in bounding_boxes:
            x1, y1, x2, y2 = bounding_boxes[0]
            if (type(bounding_boxes) != "NoneType"):
                for i in range(len(bounding_boxes)):
                    x1, y1, x2, y2 = bounding_boxes[i]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 2)
                for i in range(len(landmarks)):
                    for p in range(landmarks[i].shape[0]):
                        cv2.circle(frame,
                                   (int(landmarks[i][p, 0]),
                                    int(landmarks[i][p, 1])),
                                   2, (0, 0, 255), -1, cv2.LINE_AA)
        _, im_arr = cv2.imencode('.jpg', frame)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        message = im_b64.decode('utf-8')
    return message
