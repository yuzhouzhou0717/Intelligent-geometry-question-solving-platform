# Generated by Django 4.2.7 on 2023-11-30 08:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0008_delete_article_alter_customuser_email_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='offer',
            name='photo',
            field=models.ImageField(default=1, max_length=2083, upload_to=''),
            preserve_default=False,
        ),
    ]
