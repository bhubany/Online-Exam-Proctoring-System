"""oeps URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views as v1

urlpatterns = [
    path('', v1.index),
    path('register',v1.register), #register page
    path('register_details',v1.register_details), #getting value from form to register
    path('user_login',v1.user_login,name="user_login"), # when clicked on login button
    path('user_login_page',v1.user_login_page,name="user_login_page"), # login form
    path('admin',v1.admin_login,name="admin_login"), # login form for admin
    path('admin_login_page',v1.admin_login_page,name="admin_login_page"), # login form for admin
    path('check_admin_login',v1.check_admin_login,name="check_admin_login"), # login form for admin
    path('admin_profile',v1.admin_profile,name="admin_profile"), # login form
    path('face_verification',v1.face_verification,name="face_verification"), # login form
    path('logout',v1.logout),
    # path('admin_logout',v1.admin_logout), #admin_logout
    path('train_module',v1.train_module, name="train_module"),
    path('face_training_success',v1.face_training_success, name="face_training_success"), #when training is successs
    path('check_user_login',v1.check_user_login),  # getting login value from form
    path('face_registration',v1.face_registration, name="face_registration"), # taking User image from frontend for verification
    path('register_face',v1.register_face, name="register_face"), # redirecting to page for face registrations
    path('verify_user_face',v1.verify_user_face, name="verify_user_face"), # redirecting to page for face registrations
    path('user_profile',v1.user_profile, name="user_profile"), # redirecting to User profile after face verification
    path('final_exam_page',v1.final_exam_page, name="final_exam_page"), # redirecting to exam page
    path('home',v1.home, name="home"), # test
    path('detectme',v1.detectme), # test
    path('test',v1.test, name="forTest"), # test
    path('savetest',v1.savetest, name="savetest"), # test
    path('deleteData',v1.deleteData, name="deleteData"), # test
    path('editData',v1.editData, name="editData"), # test
    path('camTest',v1.camTest, name="camTest"), # test
    # path('faceTest',v1.faceTest, name="faceTest"), # test
    path('test01',v1.test01, name="test01"), # test
    path('checkExamStatus',v1.checkExamStatus, name="checkExamStatus"), # Is exam started?
    path('startExam',v1.startExam, name="startExam"), # Start Exam
    path('endExam',v1.endExam, name="endExam"), # end Exam
    path('viewExam',v1.viewExam, name="viewExam"), # view Exam
    path('examineeDetails',v1.examineeDetails, name="examineeDetails"), # Start Exam
    path('getOutputValues',v1.getOutputValues, name="getOutputValues"), # Actual Processing Of Image
    path('scheduleExam',v1.scheduleExam, name="scheduleExam"), # Generate Exam

]
