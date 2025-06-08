from django.urls import path
from student import views
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('studentclick/', views.studentclick_view, name='studentclick'),
    path('student-signup/', views.student_signup_view, name='student-signup'),
    path('student-dashboard/', views.student_dashboard_view, name='student-dashboard'),
    path('student-exam/', views.student_exam_view, name='student-exam'),
    path('take-exam/<int:pk>/', views.take_exam_view, name='take-exam'),
    path('start-exam/<int:pk>/', views.start_exam_view, name='start-exam'),
    path('exam-timeout/', views.exam_timeout_view, name='exam-timeout'),
    path('calculate-marks/', views.calculate_marks_view, name='calculate-marks'),
    path('view-result/', views.view_result_view, name='view-result'),
    path('check-marks/<int:pk>/', views.check_marks_view, name='check-marks'),
    path('student-marks/', views.student_marks_view, name='student-marks'),
]
