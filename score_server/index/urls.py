from django.urls import path
from index import views

urlpatterns = [
    # 用户登录注册
    path('login/', views.login),
    path('register/', views.register),
    # 照片上传
    path('upload/imageUpload', views.image_upload),
    # 创建试卷
    path('paper/add', views.addPaper),
    # 删除试卷
    path('paper/delete', views.removePaper),
    # 设置试卷名称
    path('paper/name/update', views.setPaperName),
    # 设置试卷答案
    path('paper/answer/add', views.ans_set),
    # 教师试卷列表
    path('teacher/papers', views.showPapersForTeacher),
    # 学生试卷列表
    path('student/papers', views.showPaperForStudent),
]