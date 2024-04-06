import os
import json
from rest_framework.response import Response
from rest_framework import status
from .components.openai_connector import OpenaiConnector
from dotenv import load_dotenv

from django.views import generic
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.shortcuts import render
from api.models import Settings, CodeComparison
from . import views

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def return_internal_server_error(error=None):
    if error:
        print("return_internal_server_error: ")
        print("error: ", str(error))
    error_response = {"error": "Internal Server Error"}
    error_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return Response(error_response, status=error_code)


class SettingsView(generic.ListView):
    model = Settings
    template_name = "settings.html"


class CodeComparisonView(generic.ListView):
    model = CodeComparison
    template_name = "code_comparison.html"


def optimize_code(request):
    try:
        # get POST request data
        data = json.loads(request.body)
        print("request.POST")
        print(request.POST)
        version = data["CUDA_version"]
        performance = data["speed_rating"]
        readability = data["readability_rating"]
        code = data["original_code"]

        connector = OpenaiConnector(openai_api_key)
        response = connector.create_newchat(code, version, performance, readability)
        print("view response::", response)
        optimize_code = response["content"]

        if optimize_code.startswith("```cuda\n") and optimize_code.endswith("\n```"):
            optimize_code = optimize_code[len("```cuda\n") : -len("\n```")]

        if "error" in response:
            return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            return render(
                request,
                "code_comparison.html",
                {"original_code": code, "optimized_code": optimize_code},
            )

    except KeyError as e:
        return JsonResponse(
            {"error": f"Missing parameter: {e}"}, status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return JsonResponse(
            {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def back(request):
    return HttpResponseRedirect(reverse("settings"))
