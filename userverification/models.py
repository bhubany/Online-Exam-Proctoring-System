from django.utils.timezone import now
from pyexpat import model
from unittest.util import _MAX_LENGTH
from django.db import models


# Create your models here.

 
class User_info(models.Model):
    u_id = models.AutoField(primary_key=True)
    u_f_name = models.CharField(max_length=20)
    u_m_name = models.CharField(max_length=20)
    u_l_name = models.CharField(max_length=20)
    u_email =models.CharField(max_length=30)
    u_phone=models.CharField(max_length=10)
    u_username = models.CharField(max_length=20)
    u_pass = models.CharField(max_length=32)
    is_img_registered=models.IntegerField(default=0)
    u_role=models.IntegerField(default=1, editable=False)
    reg_date=models.DateTimeField(default=now, editable=False)
    u_facl=models.CharField(max_length=30)

class exam_details(models.Model):
    e_id = models.AutoField(primary_key=True)
    e_details = models.TextField()
    e_code = models.CharField(max_length=32)
    e_staus = models.SmallIntegerField(default=0)
    e_faculty = models.CharField(max_length=32)
    e_date=models.DateTimeField(default=now, editable=False)

class examinee_details(models.Model):
    ex_id = models.AutoField(primary_key=True)
    ex_username = models.CharField(max_length=20)
    ex_code = models.CharField(max_length=32)
    ex_staus = models.SmallIntegerField(default=0)
    ex_date=models.DateTimeField(default=now, editable=False)