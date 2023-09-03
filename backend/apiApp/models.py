
from django.db import models

# Create your models here.
class userImage(models.Model):
    username=models.CharField(max_length=100)
    email=models.EmailField(max_length=200)
    password=models.CharField(max_length=200)

class user2(models.Model):
    username=models.CharField(max_length=100)
    email=models.EmailField(max_length=200)
    password=models.CharField(max_length=1000)


class medicsUser(models.Model):
    username=models.CharField(max_length=200)
    emailID=models.EmailField(max_length=200)
    phoneNo=models.CharField(max_length=100)
    password=models.CharField(max_length=1000)
    age=models.CharField(max_length=20)
    gender=models.CharField(max_length=50)
    dob=models.CharField(max_length=100)
    weight=models.CharField(max_length=100)
    height=models.CharField(max_length=100)
    bloodgroup=models.CharField(max_length=50)
    

class user(models.Model):
    username=models.CharField(max_length=100)
    email=models.EmailField(max_length=200)
    img=models.ImageField(upload_to='images/')