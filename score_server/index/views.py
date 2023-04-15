import json
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from index.models import Student, Teacher, Paper, PaperPhoto, Problem, Answer
from utils.util import tid_maker
from datetime import datetime


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
    username = request.GET["username"]
    teacher = Teacher.objects.filter(username=username)[0]
    paper = Paper.objects.create(teacher=teacher)
    return JsonResponse({"msg": "success", "data": {"paperId": paper.id}})


@require_http_methods(["GET"])
def removePaper(request):
    pid = request.GET["paperId"]
    Paper.objects.get(id=pid).delete()
    return JsonResponse({"msg": "success"})


@require_http_methods(["POST"])
def image_upload(request):
    file_obj = request.FILES.get("upload_image")
    paper_id = request.POST["paperId"]
    name = tid_maker() + '.' + file_obj.name.split(".")[1]
    file_name = settings.MEDIA_ROOT + '/paper/' + name
    with open(file_name, "wb") as f:
        for line in file_obj:
            f.write(line)
    paper = Paper.objects.get(id=paper_id)
    PaperPhoto.objects.create(photoPath=file_name, paper=paper)
    return JsonResponse({"msg": 'success', 'data': {'url': request.build_absolute_uri("/media/paper/" + name),
                                                    'name': name}})


@require_http_methods(["POST"])
def image_delete(request):
    pass


@require_http_methods(["POST"])
def ans_set(request):
    body = json.loads(request.body)
    print(body)
    # 删除之前的答案
    paper_id = body["paperId"]
    paper = Paper.objects.get(id=paper_id)
    Problem.objects.filter(paper=paper).delete()

    # 新设置问题和答案
    answer_list = body["list"]
    for ans in answer_list:
        problem = Problem.objects.create(paper=paper)
        for a in ans:
            Answer.objects.create(problem=problem, answer=a)

    return JsonResponse({"msg": "success"})


@require_http_methods(["GET"])
def setPaperName(request):
    paper_id = request.GET["paperId"]
    paper_name = request.GET["paperName"]
    paper = Paper.objects.get(id=paper_id)
    paper.name = paper_name
    paper.save()
    return JsonResponse({"msg": "success"})


@require_http_methods(["GET"])
def showPapersForTeacher(request):
    username = request.GET["username"]
    teacher = Teacher.objects.get(username=username)
    papers = Paper.objects.filter(teacher=teacher, name__isnull=False)
    return JsonResponse(
        {"msg": "success", "papers": [{"title": paper.name,
                                       "time": datetime.fromisoformat(str(paper.created_at)).strftime(
                                           "%Y-%m-%d %H:%M"),
                                       "id": paper.id} for paper in papers]})


@require_http_methods(["GET"])
def showPaperForStudent(request):
    papers = Paper.objects.filter(name__isnull=False)

    return JsonResponse(
        {"msg": "success", "papers": [{"title": paper.name,
                                       "time": datetime.fromisoformat(str(paper.created_at)).strftime(
                                           "%Y-%m-%d %H:%M"),
                                       "id": paper.id, "teacher": paper.teacher.username}
                                      for paper in papers]})
