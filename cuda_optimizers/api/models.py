from django.db import models


class OptimizationTask(models.Model):
    code = models.TextField()
    version = models.CharField(max_length=10)
    level = models.IntegerField()

class Settings(models.Model):
    #TODO create fields for each setting 
    # ex. speed, memory
    speed = models.IntegerField()

class CodeComparison(models.Model):
    #TODO create fields?? idk which tho
    original_code = models.TextField()
