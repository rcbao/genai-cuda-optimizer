from django.db import models


class OptimizationTask(models.Model):
    code = models.TextField()
    version = models.CharField(max_length=10)
    level = models.IntegerField()

class Code(models.Model):
    original_code = models.TextField()
    optimized_code = models.TextField()

class Settings(models.Model):
    #TODO create fields for each setting 
    # ex. speed, memory
    speed = models.IntegerField()
    memory = models.IntegerField()
    security = models.IntegerField()
    readability = models.IntegerField()
    code = models.ForeignKey(Code, on_delete=models.CASCADE)


class CodeComparison(models.Model):
    code = models.ForeignKey(Code, on_delete=models.CASCADE)
