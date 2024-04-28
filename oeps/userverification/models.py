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
    reg_date=models.DateTimeField(default=now, editable=False)
