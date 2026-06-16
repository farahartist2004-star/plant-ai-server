from flask import Flask, request, jsonify
import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

app = Flask(__name__)

# 1. تحميل موديل TF Lite الذكي والخفيف وإعداد الـ Interpreter
MODEL_PATH = "mint_model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# الحصول على تفاصيل طبقات المدخلات والمخرجات
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. تحميل ملفات الأسماء والبيانات للـ 4 كلاسات
with open("class_names.json") as f:
    class_names = json.load(f)

with open("plant_health_data.json") as f:
    plant_info = json.load(f)

# دالة تجهيز الصورة (الموديل الجديد بيعمل الـ Preprocessing داخلياً فبنظبط الحجم والنوع بس)
def preprocess(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

@app.route("/")
def home():
    return "AI Mint-TFLite Server Running Perfectly on Free Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # قراءة الصورة وتحويلها
    img = Image.open(file).convert("RGB")
    img_data = preprocess(img)

    # تشغيل التنبؤ (Prediction) عبر الـ TFLite Interpreter
    interpreter.set_tensor(input_details[0]['index'], img_data)
    interpreter.invoke()
    
    # الحصول على النتيجة (الاحتماليات لكل كلاس)
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # تحديد الكلاس صاحب أعلى نسبة توقع
    idx = np.argmax(prediction)
    class_name = class_names[idx]
    confidence = float(prediction[idx] * 100)

    # جلب التقرير والعلاج من ملف الـ JSON
    info = plant_info.get(class_name, {})

    return jsonify({
        "class": class_name,
        "confidence": round(confidence, 2),
        "issue": info.get("issue", "No background data available"),
        "treatment": info.get("treatment", "No treatment data available")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
