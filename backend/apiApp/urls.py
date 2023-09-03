from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from apiApp import views

urlpatterns=[
    path('user/',views.user_list),
    path('userDetails/<int:pk>/',views.user_detail),
    path('classUser/',views.userClass.as_view()),
    path('register/',views.register.as_view()),
    path('login/',views.login.as_view()),
    path('otp/',views.otp.as_view()),
    path('upload/',views.uploadImage.as_view()),
    path('image-response/',views.image_response_view,),
    path('classUser/<int:pk>/',views.userClassdetailed.as_view())
]

urlpatterns=format_suffix_patterns(urlpatterns)