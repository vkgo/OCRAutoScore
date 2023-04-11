from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from index.models import Student, Teacher


@require_http_methods(["POST"])
def login(request):
    response = {}
    username = request.POST["username"]
    identity = request.POST["identity"]
    password = request.POST["password"]
    print(request.POST)
    if identity == 'student':
        student = Student.objects.filter(username=username, password=password)
        if not student:
            response['msg'] = '用户名或密码错误'
        else:
            response['msg'] = 'success'
    else:
        teacher = Teacher.objects.filter(username=username, password=password)
        if not teacher:
            response['msg'] = '用户名或密码错误'
        else:
            response['msg'] = 'success'
    return JsonResponse(response)


@require_http_methods(["POST"])
def register(request):
    response = {}
    username = request.POST["username"]
    identity = request.POST["identity"]
    password = request.POST["password"]
    email = request.POST["email"]
    school = request.POST["school"]
    if identity == 'student':
        student = Student.objects.filter(username=username, email=email)
        if student:
            response["msg"] = "用户名或者邮箱已经被注册过"
            return JsonResponse(response)
        Student.objects.create(username=username, password=password, email=email, school=school)
    else:
        teacher = Teacher.objects.filter(username=username, password=password)
        if teacher:
            response["msg"] = "用户名或者邮箱已经被注册过"
            return JsonResponse(response)
        Teacher.objects.create(username=username, password=password, email=email, school=school)
    response["msg"] = "success"
    return JsonResponse(response)


@require_http_methods(["GET"])
def addPaper(request):
    pic = request.FILES['pic']
    save_path = '%s/booktest/%s' % (settings.MEDIA_ROOT, pic.name)
    with open(save_path, 'wb') as f:
        # 获取上传文件的内容并写到创建文件中
        # pic.chunks():分块的返回文件
        for content in pic.chunks():
            f.write(content)
    pass
