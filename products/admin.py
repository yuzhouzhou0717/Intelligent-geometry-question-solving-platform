from django.contrib import admin
from django.contrib.auth.admin import UserAdmin


from .models import Product, Offer, Cart, New, Exercise

class OfferAdmin(admin.ModelAdmin):
    list_display = ('code','discount')


class ProductAdmin(admin.ModelAdmin):
    list_display = ('name','stem','image')

class CartAdmin(admin.ModelAdmin):
    list_display = ('name','stem','image')

class NewAdmin(admin.ModelAdmin):
    list_display = ('name','stem','answer','image')

class ExerciseAdmin(admin.ModelAdmin):
    list_display = ('name', 'content' , 'image')

admin.site.register(Offer,OfferAdmin)
admin.site.register(Product,ProductAdmin)
admin.site.register(Cart,CartAdmin)
admin.site.register(New,NewAdmin)
admin.site.register(Exercise,ExerciseAdmin)
