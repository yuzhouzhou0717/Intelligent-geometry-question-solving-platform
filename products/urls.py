from django.contrib.auth.views import PasswordChangeView
from django.urls import path, include
from . import views
from products.views import product_detail_view, user_register, copy_data, test,  delete_ex_item,delete_cart_item, delete_all_cart_items, \
    search, user_profile, change_password, CustomPasswordChangeView, CustomPasswordChangeDoneView
from django.views.generic import TemplateView
from .views import user_login
from django.contrib.auth import views as auth_views
urlpatterns = [
    path('', views.index, name='home'),

    path('products/<int:product_id>/', views.product_detail_view, name='product_detail_with_id'),

    path('login/', user_login, name='login'),
    path('register/', user_register, name='register'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('copy_data/<int:product_id>/', copy_data, name='copy_data'),
    path('cart/', views.cart, name='cart'),
    path('cart/delete/<int:cart_id>/', delete_cart_item, name='delete_cart_item'),
    path('test/delete/<int:ex_id>/', delete_ex_item, name='delete_ex_item'),
    path('delete_all_cart_items/', delete_all_cart_items, name='delete_all_cart_items'),
    path('search/', search, name='search'),
    path('user_profile/', user_profile, name='user_profile'),
    path('change_password/', change_password, name='change_password'),
    path('password_change/', PasswordChangeView.as_view(template_name='password_change.html'), name='password_change'),
    path('password_change_done/', CustomPasswordChangeDoneView.as_view(), name='password_change_done'),
    path('execute_code/', views.execute_code, name='execute_code'),
    path('test/', views.test, name='test'),
]
