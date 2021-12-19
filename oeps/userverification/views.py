from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
import base64

# Create your views here.

def index(request):
    return render(request,'userverification/index.html')

# registration form
def register(request):
    return render(request,'userverification/register.html')

# Getting values from form to register
def register_details(request):
    reg_f_name=str(request.POST['r_first_name'])
    reg_m_name=str(request.POST['r_middle_name'])
    reg_l_name=str(request.POST['r_last_name'])
    reg_email=str(request.POST['r_email'])
    reg_username=str(request.POST['r_username'])
    reg_pwd=str(request.POST['r_pwd'])
    reg_conf_pwd=str(request.POST['r_conf_pwd'])

    success_message = "You have been registered successfully try loging"
    failure_message = "Error Occurs Try again later"
    context = {
        'success_msg': success_message,
        'failure_msg': failure_message,
    }
    print(reg_f_name+reg_m_name+reg_l_name+reg_email+reg_username+reg_pwd+reg_conf_pwd)
    return render(request,'userverification/face_verification.html',context)

def user_face_verificatio(request):
    img = request.POST['user_image']
    img = img.replace('data:image/png;base64,','')
    img = img.replace(' ','+')
    # img=base64.b64decode(img)
    with open('static/user_image/dataset/decoded_image.png', 'wb') as file_to_save:
        decoded_image_data = base64.b64decode(img)
        file_to_save.write(decoded_image_data)
    # print(img)
    return render(request, 'userverification/face_verification.html')

# Login Form
def login(request):
    return render(request,'userverification/login.html')

# Checking Login Credentials obtained from login form
def check_login(request):
    test_username = ['student@gmail.com','student1']
    test_pwd = "student"
    username = str(request.POST['login_username'])
    pwd = str(request.POST['login_password'])

    if ((username == test_username[0] or username == test_username[1]) and pwd == test_pwd):
        return render(request, 'userverification/user_profile.html')
    else:
        print(username+pwd)
        return render(request, 'userverification/login.html')

