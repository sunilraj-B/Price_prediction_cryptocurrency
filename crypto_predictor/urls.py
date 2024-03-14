# crypto_predictor/urls.py
from django.urls import path
from .views import update, select_crypto, predict_crypto_prices,table

app_name = 'crypto_predictor'

urlpatterns = [
    path('update/', update, name='update'),
    path('table/',table,name='table'),
    path('select_crypto/', select_crypto, name='select_crypto'),
    path('predict_crypto_prices/', predict_crypto_prices, name='predict_crypto_prices'),
    # other patterns
]
