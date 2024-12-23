# forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib.auth import get_user_model

from products.models import CustomUser


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = ['username', 'email', 'password1', 'password2']


class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(max_length=254, label='Username or Email')


class CustomPasswordChangeForm(PasswordChangeForm):
    class Meta:
        model = CustomUser  # 将 YourUserModel 替换为你的用户模型
        fields = ['old_password', 'new_password1', 'new_password2']

