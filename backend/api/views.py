import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import OptimizationTaskSerializer
from .components.openai_connector import OpenaiConnector
from dotenv import load_dotenv

from django.views import generic
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.urls import reverse
from api.models import Settings, CodeComparison, Code

from . import views

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def return_internal_server_error():
    error_response = {"error": "Internal Server Error"}
    error_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return Response(error_response, status=error_code)


class OptimizeCUDAView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = OptimizationTaskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        if not serializer or not serializer.validated_data:
            print("Invalid serializer")
            return return_internal_server_error()

        code = serializer.validated_data["code"]
        version = serializer.validated_data["version"]
        optimization_level = serializer.validated_data["level"]

        try:
            connector = OpenaiConnector(openai_api_key)
            response = connector.create_newchat(code, version, optimization_level)

            return Response(response, status=status.HTTP_200_OK)

        except ValueError as ve:
            return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(e)
            return return_internal_server_error()

class SettingsView(generic.ListView):
    model = Settings
    template_name = "settings.html"

class CodeComparisonView(generic.ListView):
    model = CodeComparison
    template_name = "code_comparison.html"

# METHODS
    
def optimize_code(request):
<<<<<<< Updated upstream
    # get POST request data
    req_speed = request.POST['speed_rating']
    req_memory = request.POST['memory_rating']
    req_security = request.POST['security_rating']
    req_readability = request.POST['readability_rating']
    req_code = request.POST['original_code']

    # create Code database object and set original code  
    code_object = Code() 
    code_object.original_code = req_code
    code_object.save()

    # create Settings database object and set data
    settings_object = Settings()
    settings_object.speed = req_speed
    settings_object.memory = req_memory
    settings_object.security = req_security
    settings_object.readability = req_readability
    settings_object.code = code_object
    settings_object.save()

    return HttpResponseRedirect(reverse("code_comparison"))
=======
    return HttpResponseRedirect(reverse("code_comparison"))

def back(request):
    return HttpResponseRedirect(reverse("settings"))
>>>>>>> Stashed changes
