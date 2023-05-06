from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=30, unique=True)
    email = models.CharField(max_length=30, unique=True)
    password = models.CharField(max_length=30)
    school = models.CharField(max_length=30)

    class Meta:
        abstract = True


class Student(User):
    pass


class Teacher(User):
    pass


class Paper(models.Model):
    teacher = models.ForeignKey(to=Teacher, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=50)


class Problem(models.Model):
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)
    type = models.CharField(max_length=3)


class Answer(models.Model):
    problem = models.ForeignKey(to=Problem, on_delete=models.CASCADE)
    answer = models.TextField()


class Score(models.Model):
    student = models.ForeignKey(to=Problem, on_delete=models.CASCADE)
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)


class UploadPhoto(models.Model):
    photoPath = models.CharField(max_length=100)
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)

    class Meta:
        abstract = True


class PaperPhoto(UploadPhoto):
    pass


class StudentUploadAnswerPhoto(UploadPhoto):
    student = models.ForeignKey(to=Student, on_delete=models.CASCADE)
