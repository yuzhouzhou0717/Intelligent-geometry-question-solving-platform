# products/models.py
from django.contrib.auth.models import User, AbstractUser, Permission, Group
from django.db import models
from django.urls import reverse

from django.db import models
class Product(models.Model):
    name = models.CharField(max_length=255, unique=True)
    stem = models.CharField(max_length=2083, default=1)
    image = models.ImageField(max_length=2083)

    def get_absolute_url(self):
        return reverse('product_detail_with_id', args=[str(self.id)])



class Cart(models.Model):
    name = models.CharField(max_length=255, unique=True)
    stem = models.CharField(max_length=2083, default=1)
    image = models.ImageField(max_length=2083)

    def get_absolute_url(self):
        return reverse('product_detail_with_id', args=[str(self.id)])


class New(models.Model):
    name = models.CharField(max_length=255, unique=True)
    stem = models.CharField(max_length=2083, default=1)
    answer = models.CharField(max_length=255, default=1)
    image = models.ImageField(max_length=2083)

    def get_absolute_url(self):
        return reverse('product_detail_with_id', args=[str(self.id)])


class Offer(models.Model):
    code = models.CharField(max_length=10)
    description = models.CharField(max_length=255)
    discount = models.FloatField()
    photo = models.ImageField(max_length=2083)


from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    groups = models.ManyToManyField(Group, related_name='custom_user_groups')
    user_permissions = models.ManyToManyField(
      Permission, related_name='custom_user_user_permissions'
    )





class Exercise(models.Model):
    name = models.CharField(max_length=100)
    content = models.TextField()
    image = models.ImageField(upload_to='exercise_images/')
