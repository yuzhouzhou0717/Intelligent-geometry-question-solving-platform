# Generated by Django 4.2.7 on 2023-11-21 13:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0003_product_stock_alter_product_price_alter_product_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Cart',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('PRICE', models.CharField(default=1, max_length=2083)),
                ('stock', models.CharField(default=1, max_length=255)),
                ('image_url', models.CharField(max_length=2083)),
            ],
        ),
    ]
