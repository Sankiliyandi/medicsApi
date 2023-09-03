#from telnetlib import STATUS
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from rest_framework.parsers import JSONParser
from .models import user2
from .serializer import *
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from django.core.serializers import serialize
import random
from .otphandler import *
from django.http import FileResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Create your views here.
# 7734154977e9d697a9b001278db1cb00e5292b7a

@csrf_exempt
def user_list(request):
    # try:
    #     snippet = user.objects.get(pk=pk)
    # except user.DoesNotExist:
    #     return HttpResponse(status=404)

    if request.method =="GET":
        users=user2.objects.all()
        serialize=userSerializer(users,many=True)
        return JsonResponse(serialize.data,safe=False)
    elif request.method =="POST":
        data=JSONParser().parse(request)
        serialize=userSerializer(data=data)

        if serialize.is_valid():
            serialize.save()
            return JsonResponse(serialize.data,status=201)
        return JsonResponse(serialize.errors,status=400)


@csrf_exempt
def user_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        users = user2.objects.get(pk=pk)
    except users.DoesNotExist:
        return HttpResponse(status=404)

    if request.method =="GET":
        #users=user2.objects.all()
        serialize=userSerializer(users)
        return JsonResponse(serialize.data,safe=False)
    # elif request.method =="POST":
    #     data=JSONParser().parse(request)
    #     serialize=userSerializer(data=data)

    #     if serialize.is_valid():
    #         serialize.save()
    #         return JsonResponse(serialize.data,status=201)
    #     return JsonResponse(serialize.errors,status=400)

    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = userSerializer(users, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        user2.delete()
        return HttpResponse(status=204)

class register(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]
    def post(self,request,formate=None):
        username=request.POST.get('username')
        email=request.POST.get('email')
        passWord=request.POST.get('password')
        print(email)
        print(passWord)
        #serialize=medicsSerializer(data=request.data)
        if medicsUser.objects.filter(emailID=email, password=passWord):
            userLog= medicsUser.objects.filter(emailID=email, password=passWord).only()
            userJSON=serialize("json",userLog)
            #serialze=medicsSerializer(email)
            return Response(userJSON,status=status.HTTP_201_CREATED)
        return Response("error",status=status.HTTP_400_BAD_REQUEST)

class login(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]
    def post(self,request,formate=None):
       
        serialize=medicsSerializer(data=request.data)
        if serialize.is_valid():
            phno=request.POST.get('phoneNo')
            rcode=random.randint(1000,9999)
            send_otp_to_phone(phno,rcode)
            request.session['otp']=rcode
            #request.session['se']=serialize
            serialize.save()
            return Response('success',status=status.HTTP_201_CREATED)
        return Response(serialize.errors,status=status.HTTP_400_BAD_REQUEST)
class otp(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]
    def post(self,request,formate=None):
        otpcode= request.session['otp']
        #serialize=request.session['se']
        postotp=request.POST.get('otp')
        if postotp==otpcode:
            serialize.save()
            return Response('success',status=status.HTTP_201_CREATED)
        return Response('errors',status=status.HTTP_400_BAD_REQUEST)
# def otp(request):
#     rcode=random.randint(1000,9999)
#     request.session['otp']=rcode
#     pass
class uploadImage(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]
    def post(self,request,format=None):
        
        serialize=userImageSerializer(data=request.data)
        if serialize.is_valid():
         #   img = open('media/hello.jpg', 'rb')

          #  response = FileResponse(img)
            return Response("response",status=status.HTTP_200_OK)
        return Response('errors',status=status.HTTP_400_BAD_REQUEST) 
class userClass(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]
    def get(self,request,formate=None):
        users=user2.objects.all()
        serialze=userSerializer(users,many=True)
        return Response(serialze.data)
    
    def post(self,request,formate=None):
        serialize=userSerializer(data=request.data)
        if serialize.is_valid():
            serialize.save()
            return Response(serialize.data,status=status.HTTP_201_CREATED)
        return Response(serialize.errors,status=status.HTTP_400_BAD_REQUEST)
def image_response_view(request):
    # Retrieve the image file
    image_path = 'segment.jpeg'  # Replace with the actual path to your image
    with default_storage.open(image_path, 'rb') as f:
        image_file = f.read()

    # Create a response with the image content
    response = HttpResponse(content_type='image/jpeg')
    response['Content-Disposition'] = 'attachment; filename="segment.jpeg"'
    response.write(image_file)

    return response

class userClassdetailed(APIView):
    authentication_class=[TokenAuthentication]
    permission_class=[IsAuthenticated]

    def get_object(self,pk):
        try:
            return user2.objects.get(pk=pk)
        except user2.DoesNotExist:
            return Http404

    def get(self,request,pk,formate=None):
        user=self.get_object(pk)
        serialize=userSerializer(user)
        return Response(serialize.data)
    def put(self,request,pk,formate=None):
        user=self.get_object(pk)
        serialize=userSerializer(user,data=request.data)
        if serialize.is_valid():
            serialize.save()
            return Response(serialize.data)
        return Response(serialize.errors,status=status.HTTP_400_BAD_REQUEST)
    def delete(self,pk,formate=None):
        user=self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)