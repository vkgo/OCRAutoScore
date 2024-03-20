import json
import os
from datetime import datetime
import PIL.Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from index.models import Student, Teacher, Paper, PaperPhoto, Problem, Answer, StudentUploadAnswerPhoto
from utils.util import tid_maker
import sys
sys.path.append('..')
from score import scoresystem


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


def image_upload(file_obj, folder):
    name = tid_maker() + '.' + file_obj.name.split(".")[1]
    file_name = settings.MEDIA_ROOT + '/{}/'.format(folder) + name
    with open(file_name, "wb") as f:
        for line in file_obj:
            f.write(line)
    return {"filename": file_name, "name": name}


@require_http_methods(["POST"])
def paper_image_upload(request):
    file_obj = request.FILES.get("upload_image")
    name_obj = image_upload(file_obj, "paper")
    print(name_obj)
    paper_id = request.POST["paperId"]
    paper = Paper.objects.get(id=paper_id)
    PaperPhoto.objects.create(photoPath="/paper/" + name_obj["name"], paper=paper)
    return JsonResponse({"msg": 'success', 'data': {
        'url': request.build_absolute_uri("/media/paper/" + name_obj["name"]), 'name': name_obj["name"]}})


@require_http_methods(["POST"])
def student_image_upload(request):
    file_obj = request.FILES.get("upload_image")
    name_obj = image_upload(file_obj, "studentAns")
    paper_id = request.POST["paperId"]
    username = request.POST["username"]
    paper = Paper.objects.get(id=paper_id)
    student = Student.objects.get(username=username)
    StudentUploadAnswerPhoto.objects.create(photoPath="/studentAns/" + name_obj["name"], paper=paper, student=student)
    return JsonResponse({"msg": 'success', 'data': {
        'url': request.build_absolute_uri("/media/studentAns/" + name_obj["name"]), 'name': name_obj["name"]}})


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
    answer_list = body["answerList"]
    type_list = body["typeList"]

    for ans_index in range(len(answer_list)):
        problem_type = type_list[ans_index]
        problem = Problem.objects.create(paper=paper, type=problem_type)
        for a in answer_list[ans_index]:
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


@require_http_methods(["GET"])
def showPaperDetail(request):
    root_url = request.scheme + '://' + request.get_host()
    paper_id = request.GET["paperId"]
    paper = Paper.objects.get(id=paper_id)
    photos = PaperPhoto.objects.filter(paper=paper)
    return JsonResponse({"msg": "success", "paperImages": [
        {"url": root_url + settings.MEDIA_URL + "paper/" + photo.photoPath.split("/")[-1], "uid": photo.id}
        for photo in photos
    ]})


@require_http_methods(["GET"])
def showPaperAnsDetail(request):
    root_url = request.scheme + '://' + request.get_host()
    paper_id = request.GET["paperId"]
    username = request.GET["username"]
    student = Student.objects.get(username=username)
    paper = Paper.objects.get(id=paper_id)
    photos = StudentUploadAnswerPhoto.objects.filter(paper=paper, student=student)
    return JsonResponse({"msg": "success", "answerImages": [
        {"url": root_url + settings.MEDIA_URL + "studentAns/" + photo.photoPath.split("/")[-1], "uid": photo.id}
        for photo in photos
    ]})


@require_http_methods(["GET"])
def deletePaperAnsPhoto(request):
    answer_id = request.GET["answerId"]
    photo = StudentUploadAnswerPhoto.objects.get(id=answer_id)
    if photo:
        try:
            os.remove(photo.photoPath)
            photo.delete()
            return JsonResponse({"msg": "success"})
        except Exception as e:
            return JsonResponse({"msg": str(e)})
    else:
        return JsonResponse({"msg": "fail"})


@require_http_methods(["GET"])
def getScore(request):
    paper_id = request.GET["paperId"]
    username = request.GET["username"]
    student = Student.objects.get(username=username)
    paper = Paper.objects.get(id=paper_id)
    photos = StudentUploadAnswerPhoto.objects.filter(paper=paper, student=student)
    print(photos)
    problems = Problem.objects.filter(paper=paper)
    answers_list = []
    for problem in problems:
        answer_obj_list = Answer.objects.filter(problem=problem)
        answers = {'section': problem.type, 'value': []}
        for a in answer_obj_list:
            answers['value'].append(a.answer)
        answers_list.append(answers)
    # 调用模型
    s = scoresystem()
    s.set_answer(answers_list)
    scores = []
    for photo in photos:
        img = PIL.Image.open(settings.MEDIA_ROOT + photo.photoPath)
        total_result = s.get_score(img)
        print(total_result)
        scores.append(total_result)
    return JsonResponse({"msg": "success", "score": scores})

