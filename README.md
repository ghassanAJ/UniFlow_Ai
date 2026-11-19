# UniFlow AI — University RAG

FastAPI + LlamaIndex + ChromaDB + **SQLite** + Ollama  
يعمل على **الويب** و**الموبايل** (متصفح الهاتف على نفس الواي فاي).

## تشغيل (Web + Mobile)

```powershell
cd C:\Users\ASUS\Desktop\UniFlow-AI
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

أو:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

- **ويب على الجهاز:** http://127.0.0.1:8000/
- **موبايل:** من الهاتف افتح `http://IP-الكمبيوتر:8000/` (نفس الواي فاي)
- روابط الموبايل تظهر أعلى الداشبورد وبعد `/health`

### معرفة IP الكمبيوتر (Windows)
```powershell
ipconfig
```
خذ عنوان IPv4 لشبكة Wi-Fi (مثال `192.168.1.10`) ثم من الموبايل:
`http://192.168.1.10:8000/`

إذا الموبايل ما بفتح: اسمح لـ Python في جدار حماية Windows عند أول تشغيل، أو أضف قاعدة للمنفذ 8000.

## متغيرات اختيارية (`.env`)
```
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=*
```
`CORS_ORIGINS=*` يسمح لتطبيق Flutter/Web خارجي ينادي الـ API.

## ربط Flutter / Laravel / Web
قاعدة الـ API:
```
http://<host>:8000
```
Endpoints:
- `PUT /students/{id}`
- `PUT /courses/{id}`
- `POST /ingest`
- `POST /chat`
- `GET /health`

## مهم لنجاح الشات
1. ارفع المحاضرة بـ `course_id` مثل `CS101`
2. سجّل الطالب بنفس `course_id`
3. بعد تعديل الكود: أعد تشغيل السيرفر
