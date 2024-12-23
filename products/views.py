from django.views.decorators.csrf import csrf_exempt
import subprocess

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import PasswordChangeView
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib import messages

from .forms import CustomAuthenticationForm, CustomUserCreationForm, CustomPasswordChangeForm
from .models import Product, Cart, New, Exercise
from django.db import IntegrityError
import os
import django
from django.http import HttpResponse
from .import api2  # 导入api2模块
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject2.settings")
django.setup()

def copy_data(request, product_id):
    try:
        product_object = get_object_or_404(Product, pk=product_id)

        # 在第二个数据库中检查是否已存在
        if Cart.objects.filter(name=product_object.name).exists():
            return JsonResponse({'message': '该题目已添加'})

        # 创建一个新的对象，并将属性复制
        cart_object = Cart(
            name=product_object.name,
            stem=product_object.stem,
            image=product_object.image,
        )

        # 保存到第二个数据库
        cart_object.save()

        # 返回 JSON 数据表示复制成功
        return JsonResponse({'message': '添加成功'})
    except Product.DoesNotExist:
        # 处理第一个数据库中对象不存在的情况
        return JsonResponse({'error': '未找到'}, status=404)
    except IntegrityError:
        # 处理在第二个数据库中重复插入的情况
        return JsonResponse({'message': '出题库中已有该题目'})
    except Exception as e:
        # 处理其他异常，例如复制数据时出错
        return JsonResponse({'error': str(e)}, status=500)


def index(request):
    products = Product.objects.all()
    return render(request, 'index.html',
                  {'products': products})


def cart(request):
    # 获取第二个数据库的所有记录
    cart_items = Cart.objects.all()
    return render(request, 'cart.html', {'cart_items': cart_items})


def test(request):
    # 获取第二个数据库的所有记录
    ex_items = Exercise.objects.all()
    return render(request, 'test.html', {'ex_items': ex_items})


def new(request):
    # 获取第二个数据库的所有记录
    new_items = New.objects.all()
    return render(request, 'new_exercises/new.html', {'new_items': new_items})


def delete_cart_item(request, cart_id):
    cart_item = get_object_or_404(Cart, pk=cart_id)
    cart_item.delete()
    return redirect('cart')


def delete_ex_item(request, ex_id):
    ex_item = get_object_or_404(Exercise, pk=ex_id)
    ex_item.delete()
    return redirect('test')


def delete_all_cart_items(request):
    # 删除所有 Cart 记录
    Cart.objects.all().delete()
    return redirect('cart')


def product_detail_view(request, product_id):
    product = get_object_or_404(Product, id=product_id)

    return render(request, 'product_detail.html', {'product': product})


def user_login(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')  # 重定向到登录后的页面
        else:
            messages.error(request, '用户名或密码输入有误，请重试。')
    else:
        form = CustomAuthenticationForm()

    return render(request, 'login.html', {'form': form})


def user_register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')  # 重定向到注册后的页面
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


@login_required
def user_profile(request):
    return render(request, 'user_profile.html')


@login_required
def change_password(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # 重定向到修改密码成功的页面
    else:
        form = CustomPasswordChangeForm(user=request.user)
    return render(request, 'change_password.html', {'form': form})


class CustomPasswordChangeView(PasswordChangeView):
    form_class = CustomPasswordChangeForm
    success_url = '/login/'


from django.views.generic import TemplateView


class CustomPasswordChangeDoneView(TemplateView):
    template_name = 'login.html'

    def get(self, *args, **kwargs):
        # 调用父类的 get 方法处理密码修改成功的逻辑
        response = super().get(*args, **kwargs)

        # 重定向到登录页面
        return redirect('login')


def search(request):
    query = request.GET.get('q')

    if query:
        results = Product.objects.filter(stem__icontains=query)
    else:
        results = []

    return render(request, 'search_results.html', {'results': results, 'query': query})
from django.views.generic import TemplateView


def execute_code(request):
    # 调用api2模块中的execute_code函数
    api2.execute_code(request)

    return HttpResponse("Code executed successfully")
