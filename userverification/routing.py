from django.urls import re_path
from .views import WSConsumer, WSAdmin


ws_urlpatterns=[
    re_path(r'ws/some_url',WSConsumer.as_asgi()),
    re_path(r'ws/admin',WSAdmin.as_asgi()),
]
