# Air Write — Hand Gesture Drawing on Screen

تطبيق Computer Vision يتيح لك الكتابة والرسم على الشاشة باستخدام إصبعك فقط، بدون ماوس أو لوحة مفاتيح أو أي جهاز لمس. الكاميرا تتبع يدك، وتتحول حركة إصبعك إلى خط مرسوم على الشاشة (Air Writing / Virtual Pen).

---

## 1. الفكرة الأساسية (Concept)

1. الكاميرا تلتقط فيديو لايف من المستخدم.
2. نموذج Hand Tracking (MediaPipe Hands) يكتشف اليد و21 نقطة (landmarks) على الأصابع.
3. نحدد موقع رأس السبابة (Index Fingertip) كـ "قلم".
4. نحدد الإيماءة (Gesture) لتفعيل/تعطيل الرسم:
   - **إصبع واحد مرفوع (سبابة فقط)** → وضع الرسم (Draw Mode).
   - **إصبعين مرفوعين (سبابة + وسطى)** → وضع التحريك بدون رسم (Move/Hover Mode).
   - **قبضة مغلقة (Fist)** → إيقاف/Pause.
   - **كف مفتوح بالكامل (5 أصابع)** → مسح الشاشة (Clear Canvas).
5. نقطة الإصبع تُسجَّل كمسار (path) ويتم رسمها على طبقة Canvas شفافة فوق الفيديو.
6. النتيجة تظهر كأن المستخدم يكتب فعليًا في الهواء.

---

## 2. المزايا (Features)

- ✏️ رسم وكتابة حرة بالهواء (Air Drawing).
- ✋ نظام Gesture Recognition للتحكم بدون لمس الشاشة.
- 🎨 اختيار لون وحجم الفرشاة من شريط أدوات يظهر على الشاشة (Virtual Toolbar).
- 🧽 مسح كامل أو مسح جزئي (Eraser gesture).
- 💾 حفظ الرسمة كصورة PNG.
- ⏪ Undo / Redo للخطوط الأخيرة.
- 🖥️ يعمل Real-time (30+ FPS) على معالج عادي بدون GPU.
- 🔡 (اختياري) تحويل الكتابة المرسومة إلى نص عبر OCR.
- 🌐 خيار تشغيله كتطبيق Desktop (Python) أو Web (Browser, بدون تثبيت).

---

## 3. اختيار التقنية (Tech Stack)

### الخيار الأول — Python (الأسهل والأسرع للتطوير)
| الغرض | المكتبة |
|---|---|
| التقاط الفيديو | OpenCV (`opencv-python`) |
| تتبع اليد والأصابع | MediaPipe (`mediapipe`) |
| الرسم والعمليات الحسابية | NumPy |
| واجهة المستخدم (اختياري) | OpenCV windows أو PyQt/Tkinter |
| حفظ الصور | OpenCV / Pillow |
| OCR (اختياري) | Tesseract (`pytesseract`) |

### الخيار الثاني — Web (يعمل من المتصفح مباشرة، بدون تثبيت)
| الغرض | المكتبة |
|---|---|
| تتبع اليد | MediaPipe Hands (JS) أو TensorFlow.js Handpose |
| الواجهة | Next.js / React + Canvas API |
| الرسم | HTML5 Canvas |
| التنسيق | Tailwind CSS |

> الأفضل تبدأ بنسخة Python لأنها أسرع للتجربة والبرمجة، وبعدها ممكن تحوّلها لنسخة Web إذا حبيت تنشرها أونلاين.

---

## 4. هيكلية المشروع (Project Structure) — نسخة Python

```
air-write/
│
├── main.py                     # نقطة تشغيل المشروع
├── requirements.txt            # المكتبات المطلوبة
├── README.md
│
├── config.py                   # ثوابت وإعدادات (ألوان، حجم الفرشاة، عتبات الإيماءات)
│
├── core/
│   ├── __init__.py
│   ├── camera.py                # فتح الكاميرا وقراءة الفريمات
│   ├── hand_tracker.py          # كلاس HandTracker: تشغيل MediaPipe واستخراج landmarks
│   ├── gesture_recognizer.py    # تحديد نوع الإيماءة (Draw / Move / Clear / Pause)
│   ├── finger_utils.py          # دوال مساعدة: حساب المسافة بين النقاط، تحديد الأصابع المرفوعة
│   └── smoothing.py             # تنعيم حركة النقطة (لتقليل الرعشة/الجيتر)
│
├── drawing/
│   ├── __init__.py
│   ├── canvas.py                 # كلاس DrawingCanvas: يحتفظ بطبقة الرسم ويدمجها مع الفيديو
│   ├── brush.py                  # إدارة حجم/لون/شكل الفرشاة
│   └── history.py                # Undo/Redo (تخزين نسخ سابقة من الـ canvas)
│
├── ui/
│   ├── __init__.py
│   ├── toolbar.py                # شريط الأدوات الافتراضي (ألوان/أحجام) المرسوم على الفيديو
│   └── overlay.py                # عرض حالة النظام (FPS, الوضع الحالي, تعليمات)
│
├── ocr/
│   └── text_recognizer.py        # (اختياري) تحويل الرسم المكتوب إلى نص عبر Tesseract
│
├── assets/
│   └── icons/                    # أيقونات الأدوات (فرشاة، ممحاة، حفظ...)
│
├── saved_drawings/                # مجلد تُحفظ فيه الصور الناتجة (PNG)
│
└── tests/
    ├── test_gesture_recognizer.py
    └── test_finger_utils.py
```

---

## 5. شرح الوحدات الأساسية (Core Modules)

### `core/hand_tracker.py`
- يستخدم `mediapipe.solutions.hands`.
- يرجّع لكل فريم: قائمة بـ 21 نقطة (x, y, z) لكل يد مكتشَفة.
- النقاط المهمة:
  - `landmark[8]` → رأس السبابة (Index Tip) = القلم.
  - `landmark[12]` → رأس الوسطى (Middle Tip).
  - `landmark[4]` → رأس الإبهام (Thumb Tip).

### `core/gesture_recognizer.py`
يحدد الإيماءة بناءً على عدد الأصابع المرفوعة (بمقارنة ارتفاع كل tip عن المفصل اللي تحته):

| الإيماءة | الأصابع المرفوعة | الحدث |
|---|---|---|
| Draw | السبابة فقط | بداية/استمرار خط جديد |
| Move/Hover | السبابة + الوسطى | تحريك بدون رسم |
| Select Color | تمرير فوق شريط الأدوات | تغيير اللون/الحجم |
| Clear All | الكف مفتوح (5 أصابع) لمدة ~1 ثانية | مسح الشاشة بالكامل |
| Pause | قبضة مغلقة | تجميد الرسم |

### `core/smoothing.py`
- متوسط متحرك (Moving Average) أو فلتر One-Euro لتنعيم حركة الإصبع.
- يمنع ظهور خطوط متقطعة أو مرتجفة بسبب ضجيج الكاميرا.

### `drawing/canvas.py`
- طبقة Numpy array بنفس أبعاد الفيديو (شفافة/سوداء).
- كل نقطة جديدة تُرسم بخط (`cv2.line`) بين الموضع السابق والحالي.
- تُدمج هذه الطبقة مع فريم الكاميرا الحالي بـ `cv2.addWeighted` أو bitwise masking.

### `ui/toolbar.py`
- شريط مستطيلات ملوّنة يُرسم في أعلى الشاشة (مثل: أحمر / أخضر / أزرق / أصفر / ممحاة).
- إذا رأس السبابة دخل ضمن منطقة مستطيل لون معيّن → يتغيّر اللون الحالي.

---

## 6. تدفق العمل (Pipeline / Flow)

```
[الكاميرا] 
    ↓ frame
[Hand Tracker / MediaPipe] 
    ↓ 21 landmark points
[Gesture Recognizer] 
    ↓ (draw / move / clear / pause)
[Smoothing Filter] 
    ↓ نقطة مُنعّمة (x, y)
[Drawing Canvas] 
    ↓ تحديث طبقة الرسم
[Overlay + Toolbar] 
    ↓ دمج كل الطبقات
[عرض على الشاشة] (cv2.imshow أو نافذة التطبيق)
```

---

## 7. خوارزمية تحديد الأصابع المرفوعة (مبسّطة)

```python
def fingers_up(landmarks):
    tips_ids = [8, 12, 16, 20]   # السبابة، الوسطى، البنصر، الخنصر
    fingers = []

    # الإبهام (منطق مختلف لأنه يتحرك جانبيًا)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # باقي الأصابع: التِب أعلى من المفصل اللي تحته بـ 2 = مرفوع
    for tip_id in tips_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers   # [إبهام, سبابة, وسطى, بنصر, خنصر]
```

---

## 8. متطلبات التشغيل (Requirements)

```
opencv-python>=4.9.0
mediapipe>=0.10.0
numpy>=1.26.0
pytesseract>=0.3.10   # اختياري - لتحويل الكتابة لنص
```

تثبيت:
```bash
pip install -r requirements.txt
```

تشغيل:
```bash
python main.py
```

---

## 9. الإعدادات القابلة للتعديل (`config.py`)

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DEFAULT_BRUSH_COLOR = (0, 0, 255)     # أحمر BGR
DEFAULT_BRUSH_THICKNESS = 6
ERASER_THICKNESS = 40

SMOOTHING_FACTOR = 0.6
CLEAR_GESTURE_HOLD_SECONDS = 1.0

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
```

---

## 10. مراحل التطوير المقترحة (Roadmap)

1. **MVP**: كاميرا + تتبع يد + رسم بإصبع واحد فقط (سبابة) بلون ثابت.
2. **إضافة الإيماءات**: Move / Clear / Pause.
3. **شريط الأدوات الافتراضي**: ألوان + أحجام + ممحاة.
4. **Undo / Redo + حفظ الصورة كـ PNG**.
5. **تنعيم الحركة (Smoothing)** لتحسين جودة الخط.
6. **(اختياري) OCR**: تحويل الكتابة المرسومة إلى نص فعلي.
7. **(اختياري) نسخة Web** بنفس المنطق عبر MediaPipe.js + Canvas API لنشره أونلاين بدون تثبيت.
8. **(اختياري) دعم يدين** (رسم بيدين أو يد للرسم ويد للتحكم بالأدوات).

---

## 11. تحديات متوقعة (Challenges)

- **الإضاءة الضعيفة** تقلل دقة اكتشاف اليد → الحل: زيادة `min_detection_confidence` أو تحسين الإضاءة.
- **ارتجاف الخط (Jitter)** → الحل: فلتر تنعيم (Smoothing/One-Euro Filter).
- **اكتشاف خاطئ لعدد الأصابع** → الحل: ضبط العتبات حسب زاوية الكاميرا، واختبار بأكثر من زاوية.
- **الأداء (FPS) منخفض على أجهزة ضعيفة** → الحل: تقليل دقة الفريم (مثلاً 640×480) أو تشغيل MediaPipe بـ `model_complexity=0`.

---

## 12. إمكانية التوسعة (Future Ideas)

- تحويل المشروع إلى لوح تفاعلي للعروض التقديمية (Presentation Pointer/Annotator).
- دمجه مع الذكاء الاصطناعي لتصحيح الكتابة اليدوية تلقائيًا.
- دعم التحكم بالأشكال (رسم دائرة/مستطيل بإيماءة معينة).
- نسخة تعليمية للأطفال لتعليم الحروف عبر الكتابة بالهواء.
