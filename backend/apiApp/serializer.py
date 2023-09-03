from dataclasses import field
from rest_framework import serializers
from apiApp.models import user2,medicsUser,user

class userSerializer(serializers.ModelSerializer):
    class Meta:
        model=user2
        fields=['username','email','password']


class medicsSerializer(serializers.ModelSerializer):
    class Meta:
        model=medicsUser
        fields=['username','emailID','phoneNo','password','age','gender','dob','weight','height','bloodgroup']
 
class userImageSerializer(serializers.ModelSerializer):
       class Meta:
          model=user
          fields=['username','email','img']
 
    #  username=serializers.CharField(max_length=100)
    #  email=serializers.EmailField(max_length=200)
    #  password=serializers.CharField

    #  def create(self, validated_data):
    #      return user.objects.create(validated_data)
    

    #  def update(self, instance, validated_data):
    #      instance.username=validated_data.get("username",instance.username)
    #      instance.email=validated_data.get("email",instance.email)
    #      instance.password=validated_data.get("password",instance.password)
    #      instance.save()
    #      return instance