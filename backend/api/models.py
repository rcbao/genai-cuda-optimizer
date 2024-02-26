from django.db import models


class OptimizationTask(models.Model):
    code = models.TextField()
    version = models.CharField(max_length=10)
    level = models.IntegerField()
