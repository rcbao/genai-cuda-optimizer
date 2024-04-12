from django.contrib import admin

# Register your models here.
from api.models import Code, Settings, CodeComparison

admin.site.register(Code)
admin.site.register(Settings)
admin.site.register(CodeComparison)