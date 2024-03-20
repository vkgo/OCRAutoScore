from django.urls import path
from index import views

urlpatterns = [
    # 用户登录注册
    path('login/', views.login),
    path('register/', views.register),
    # 照片上传
    path('upload/imageUpload', views.paper_image_upload),
    # 学生答案照片上传
    path('student/answer/imageUpload', views.student_image_upload),
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
    # 学生试卷图片获取
    path('student/paper/detail', views.showPaperDetail),
    # 学生作答试卷图片获取
    path('student/paper/answer/detail', views.showPaperAnsDetail),
    # 学生作答试卷图片删除
    path('student/paper/answer/delete', views.deletePaperAnsPhoto),
    # 获取分数
    path('student/paper/score', views.getScore)
]