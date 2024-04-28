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
    path('login',v1.login), # login form
    path('logout',v1.logout),
    path('train_module',v1.train_module, name="train_module"),
    path('check_login',v1.check_login),  # getting login value from form
    path('face_registration',v1.user_face_registration, name="face_registration"), # taking User image from frontend for verification
    path('home',v1.home), # test
    path('detectme',v1.detectme), # test
    path('test',v1.test, name="forTest"), # test
    path('savetest',v1.savetest, name="savetest"), # test
    path('deleteData',v1.deleteData, name="deleteData"), # test
    path('editData',v1.editData, name="editData"), # test
    path('camTest',v1.camTest, name="camTest"), # test
    path('faceTest',v1.faceTest, name="faceTest"), # test
]
