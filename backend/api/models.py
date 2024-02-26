from django.db import models


class OptimizationTask:
    code = models.TextField()
    version = models.CharField(max_length=10)
    level = models.IntegerField()
