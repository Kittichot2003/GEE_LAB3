# GEE_LAB3
Lab 3: Machine Learning for Land Use / Land Cover Classification
📌 Course

GEOGRAPHIC DATA SCIENCE (ภม.338)
Lab 3: Supervised & Unsupervised Machine Learning

🌏 Project Overview

โครงการนี้มีเป้าหมายเพื่อสร้างแผนที่ Land Use / Land Cover (LULC)
สำหรับพื้นที่ชุ่มน้ำและพื้นที่เกษตรกรรมในประเทศไทย โดยใช้ข้อมูลจากดาวเทียมและ Machine Learning

ข้อจำกัดสำคัญ:

มี Training Data จำกัด
ไม่สามารถเก็บ Ground Truth ได้ครบทุกพื้นที่

ดังนั้น Lab นี้จึงมุ่งตอบคำถามสำคัญ:

วิธีไหนแม่นยำที่สุดเมื่อข้อมูลฝึกมีน้อย?
การเพิ่ม Spectral Indices ช่วยจริงหรือไม่?
แผนที่ที่ได้ “เชื่อถือได้แค่ไหน”?
🧠 Task 1: Training Strategy Design
🔹 1.1 Land Cover Classes

กำหนดทั้งหมด 5 Classes ดังนี้:

Class	คำอธิบาย
Urban	พื้นที่เมือง ถนน อาคาร
Forest	ป่าไม้หนาแน่น
Water	แหล่งน้ำ เช่น แม่น้ำ บึง
Agriculture	พื้นที่เกษตรกรรม
Miscellaneous	พื้นที่อื่น เช่น ดินเปล่า

เหตุผล:
เลือก 5 class เพื่อ balance ระหว่าง complexity และความแม่นยำ
(มากเกินไป → training ยาก, น้อยเกินไป → สูญเสียรายละเอียด)

🔹 1.2 Training Data Strategy

ใช้ Hybrid Approach

วาด polygon เองจากภาพ Sentinel-2
ใช้ reference จาก:
MODIS Land Cover (MCD12Q1)
Dynamic World

เหตุผล:

ลด bias จาก dataset เดียว
เพิ่ม diversity ของ training samples
เหมาะกับกรณีข้อมูลน้อย
🔹 1.3 Train / Validation Split

ใช้ 2 วิธี:

✅ Random Split (80/20)
แบ่งข้อมูลแบบสุ่ม
ง่ายและรวดเร็ว
✅ Spatial Cross-validation
แบ่งตามพื้นที่ (grid-based)
ลดปัญหา Spatial Autocorrelation

ข้อแตกต่าง:

วิธี	ข้อดี	ข้อเสีย
Random	ง่าย	Accuracy สูงเกินจริง
Spatial	realistic	ทำยาก
🔹 1.4 Features Used
📡 Spectral Bands:
B2 (Blue)
B3 (Green)
B4 (Red)
B8 (NIR)
B11 (SWIR1)
B12 (SWIR2)
🌱 Spectral Indices:
NDVI (Vegetation)
NDWI (Water)
NDBI (Built-up)

เหตุผล:

Bands → ข้อมูลพื้นฐาน
Indices → เพิ่มการแยก class ที่คล้ายกัน
🤖 Task 2: Model Comparison
🔹 Models Used
1. Random Forest
ทดลองหลายค่า numberOfTrees (50, 100, 200)
2. Support Vector Machine (SVM)
🔹 Evaluation Metrics

ใช้มากกว่า Overall Accuracy:

Overall Accuracy
Producer’s Accuracy
User’s Accuracy
Kappa Coefficient
F1-score (per class)
🔹 Results Summary
Model	Accuracy	Kappa	Notes
Random Forest	สูง	สูง	stable
SVM	ปานกลาง	ปานกลาง	sensitive

Insight:

Random Forest ทำงานดีกว่าเมื่อ Training Data น้อย
SVM ต้อง tuning มากกว่า
🌟 Task 3: Feature Importance Analysis
🔹 Top Important Features
NDVI → แยก vegetation ได้ดีที่สุด
NDWI → แยก water
SWIR → แยก built-up vs soil

Interpretation:
สอดคล้องกับหลักทางกายภาพ:

พืชสะท้อน NIR สูง → NDVI สูง
น้ำดูดกลืน NIR → NDWI สำคัญ
🔹 Removing Low-Importance Features

ทดลองตัด feature ที่ importance ต่ำ:

Case	Accuracy
All features	สูงสุด
Remove low importance	ลดลงเล็กน้อย

Conclusion:

Feature บางตัว redundant แต่ยังช่วย model stability
⚠️ Task 4: Uncertainty Analysis
🔹 Confusion Analysis

Class ที่สับสนมากที่สุด:

Agriculture vs Forest

เหตุผล:

spectral signature คล้ายกัน
seasonal variation
🔹 Low Confidence Areas

บริเวณที่ model ไม่มั่นใจ:

ขอบพื้นที่เกษตร
พื้นที่ชุ่มน้ำผสม
🔹 Reliability Statement

แผนที่นี้:

เชื่อถือได้ในระดับ moderate to high
แม่นยำสูงใน:
น้ำ
เมือง
ไม่แน่นอนใน:
พื้นที่เกษตร/ป่า
🔍 Critical Questions
1. เพิ่ม Training Data 2 เท่า
Accuracy เพิ่มขึ้นเล็กน้อย (~3–8%)
diminishing return เมื่อ data มากขึ้น
2. Spatial Autocorrelation
ทำให้ accuracy “ดูสูงเกินจริง”
เพราะ pixel ใกล้กันมีค่าคล้ายกัน
3. Worst Performing Class

Agriculture

แนวทางแก้:

เพิ่ม training samples
เพิ่ม temporal data (time series)
ใช้ indices เพิ่มเติม
4. Transferability
สิ่งที่ต้องเปลี่ยน:
Training data
Threshold
Seasonal data
สิ่งที่ใช้ซ้ำได้:
Code pipeline
Feature engineering
Model structure
⚙️ Reproducibility
📁 Structure
Lab3/
│── notebook.ipynb
│── data/
│── outputs/
│── README.md
