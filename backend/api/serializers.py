from rest_framework import serializers
from .models import OptimizationTask


class OptimizationTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = OptimizationTask
        fields = ["code", "version", "level"]

    def validate_optimization_level(self, value):
        if not (1 <= value <= 5):
            raise serializers.ValidationError(
                "Optimization level must be between 1 and 5."
            )
        return value
