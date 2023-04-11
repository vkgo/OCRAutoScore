from django.urls import path
from index import views

urlpatterns = [
    # 用户登录注册
    path('login/', views.login),
    path('register/', views.register)
]