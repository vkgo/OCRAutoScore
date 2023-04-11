from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=30,unique=True)
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


class PaperPhoto(models.Model):
    photo = models.ImageField(upload_to="paperPhoto")
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)


class Problem(models.Model):
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)


class Answer(models.Model):
    problem = models.ForeignKey(to=Problem, on_delete=models.CASCADE)
    answer = models.TextField()


class Score(models.Model):
    student = models.ForeignKey(to=Problem, on_delete=models.CASCADE)
    paper = models.ForeignKey(to=Paper, on_delete=models.CASCADE)


class StudentUploadAnswer(PaperPhoto):
    student = models.ForeignKey(to=Student, on_delete=models.CASCADE)