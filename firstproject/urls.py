"""
URL configuration for firstproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# firstproject/urls.py
# from django.contrib import admin
# from django.urls import path, include
# from django.shortcuts import redirect

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('crypto/', include('crypto_predictor.urls')),
#     path('', lambda request: redirect('select_crypto'), name='redirect_root'),  # Updated redirection to select_crypto
# ]
# firstproject/urls.py
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from crypto_predictor.views import update

urlpatterns = [
    path('admin/', admin.site.urls),
    path('crypto/', include('crypto_predictor.urls')),
    path('update/', update, name='update'),  # Include the update URL pattern
    path('', lambda request: redirect('update'), name='redirect_root'),
]

