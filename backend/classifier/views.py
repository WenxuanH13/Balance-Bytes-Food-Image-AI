from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ml import predict_image

def index_view(request):
    return render(request, "classifier/index.html")

@api_view(['POST'])
def predict_view(request):
    if 'image' not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    label, confidence = predict_image(request.FILES['image'])
    return Response({
        "label": label,
        "confidence": confidence
    })

# Create your views here.
