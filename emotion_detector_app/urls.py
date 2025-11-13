from django.urls import path
from . import views  # Import views from the current app

urlpatterns = [
    # When someone visits the root URL, home view is used
    path('', views.home, name='home'),
]