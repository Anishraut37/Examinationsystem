from django.shortcuts import render, redirect, reverse
from . import forms, models
from django.db.models import Sum
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings
from datetime import date, timedelta, datetime
from exam import models as QMODEL
from teacher import models as TMODEL

# for showing signup/login button for student
def studentclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return render(request, 'student/studentclick.html')

def student_signup_view(request):
    userForm = forms.StudentUserForm()
    studentForm = forms.StudentForm()
    mydict = {'userForm': userForm, 'studentForm': studentForm}
    if request.method == 'POST':
        userForm = forms.StudentUserForm(request.POST)
        studentForm = forms.StudentForm(request.POST, request.FILES)
        if userForm.is_valid() and studentForm.is_valid():
            user = userForm.save()
            user.set_password(user.password)
            user.save()
            student = studentForm.save(commit=False)
            student.user = user
            student.save()
            my_student_group = Group.objects.get_or_create(name='STUDENT')
            my_student_group[0].user_set.add(user)
        return redirect('studentlogin')
    return render(request, 'student/studentsignup.html', context=mydict)

def is_student(user):
    return user.groups.filter(name='STUDENT').exists()

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_dashboard_view(request):
    dict = {
        'total_course': QMODEL.Course.objects.all().count(),
        'total_question': QMODEL.Question.objects.all().count(),
    }
    return render(request, 'student/student_dashboard.html', context=dict)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_exam_view(request):
    courses = QMODEL.Course.objects.all()
    return render(request, 'student/student_exam.html', {'courses': courses})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def take_exam_view(request, pk):
    course = QMODEL.Course.objects.get(id=pk)
    total_questions = QMODEL.Question.objects.all().filter(course=course).count()
    questions = QMODEL.Question.objects.all().filter(course=course)
    total_marks = sum(q.marks for q in questions)
    return render(request, 'student/take_exam.html', {'course': course, 'total_questions': total_questions, 'total_marks': total_marks})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def start_exam_view(request, pk):
    course = QMODEL.Course.objects.get(id=pk)
    questions = QMODEL.Question.objects.all().filter(course=course)
    response = render(request, 'student/start_exam.html', {'course': course, 'questions': questions})
    response.set_cookie('course_id', course.id)
    response.set_cookie('exam_start_time', str(datetime.now()))
    response.set_cookie('exam_duration', course.duration)  # Set duration cookie
    return response

def exam_timeout_view(request):
    return render(request, 'exam/exam_timeout.html')

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def calculate_marks_view(request):
    if request.COOKIES.get('course_id'):
        course_id = request.COOKIES.get('course_id')
        course = QMODEL.Course.objects.get(id=course_id)
        exam_start_time = request.COOKIES.get('exam_start_time')
        exam_duration = int(request.COOKIES.get('exam_duration'))
        exam_start_time = datetime.strptime(exam_start_time, '%Y-%m-%d %H:%M:%S.%f')
        if datetime.now() > exam_start_time + timedelta(minutes=exam_duration):
            return redirect('exam-timeout') # Redirect to timeout page
        total_marks = 0
        questions = QMODEL.Question.objects.filter(course=course)
        for i, question in enumerate(questions, start=1):
            selected_ans = request.COOKIES.get(str(i))
            if selected_ans == question.answer:
                total_marks += question.marks
        student = models.Student.objects.get(user_id=request.user.id)
        result = QMODEL.Result(marks=total_marks, exam=course, student=student)
        result.save()
        return redirect('view-result')

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def view_result_view(request):
    courses = QMODEL.Course.objects.all()
    return render(request, 'student/view_result.html', {'courses': courses})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def check_marks_view(request, pk):
    course = QMODEL.Course.objects.get(id=pk)
    student = models.Student.objects.get(user_id=request.user.id)
    results = QMODEL.Result.objects.filter(exam=course, student=student)
    return render(request, 'student/check_marks.html', {'results': results})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_marks_view(request):
    courses = QMODEL.Course.objects.all()
    return render(request, 'student/student_marks.html', {'courses': courses})
