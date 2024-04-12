from rest_framework import serializers
from .models import OptimizationTask


class OptimizationTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = OptimizationTask
        fields = ["code", "version", "level"]
