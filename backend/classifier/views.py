# backend/classifier/views.py

from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .ml_detector import detect_food
from .ml_classifier import classify_food

def index_view(request):
    return render(request, "classifier/index.html")

@api_view(['POST'])
def predict_view(request):
    file_obj = request.FILES.get("image")
    if not file_obj:
        return JsonResponse({"error": "No image uploaded"}, status=400)

    image_bytes = file_obj.read()

    detector = detect_food(image_bytes)
    is_food = detector["is_food"]

    guess = None        
    confidence = None   
    classifier = None   

    if is_food:
        label = "Food"
        classifier = classify_food(image_bytes)
        guess = classifier["label"]
        confidence = classifier["confidence"]
    else:
        label = "Non-Food"
        confidence = detector.get(
            "prob_non_food",
            1.0 - detector.get("prob_food", 0.0) 
        )

    return JsonResponse({
        "label": label,       
        "guess": guess,         
        "is_food": is_food,      
        "confidence": confidence,
        "detector": detector,
        "classifier": classifier,
    })
