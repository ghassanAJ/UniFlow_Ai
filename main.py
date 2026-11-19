"""
University RAG service  (FastAPI + LlamaIndex + ChromaDB + SQLite)

الفكرة:
- كل ملف محاضرة بينعمله ingest مع metadata: course_id, lecture_id, file_name
- وقت الشات: بنجيب المواد المسجلة للطالب من SQLite (مش من الكلاينت!)
  وبنفلتر البحث بالـ vector store على هاي المواد فقط -> الطالب ما بيشوف غير ملفاته.
"""
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
import pymupdf
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import db

# ----------------------------------------------------------------------------
# إعدادات (مجاني محلي عبر Ollama — ما بحتاج OPENAI_API_KEY)
# ----------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "qwen2.5:3b")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text")

Settings.llm = Ollama(
    model=OLLAMA_LLM,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    request_timeout=120.0,
)
Settings.embed_model = OllamaEmbedding(
    model_name=OLLAMA_EMBED,
    base_url=OLLAMA_BASE_URL,
)
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("lectures_ollama")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

SYSTEM_PROMPT = """أنت مساعد دراسي لطالب جامعي.
أجب فقط بالاعتماد على المقاطع المأخوذة من ملفات محاضرات الطالب الموجودة في السياق.
إذا لم تجد الإجابة في السياق قل بوضوح: "ما لقيت هاي المعلومة في ملفاتك"، ولا تخترع إجابة ولا تستخدم معلومات من خارج الملفات.
أجب بنفس لغة سؤال الطالب، وبشكل واضح ومنظم، وأشر لاسم المحاضرة/الملف عند الإمكان."""

db.init_db()

# ----------------------------------------------------------------------------
# ذاكرة المحادثة (للإنتاج استخدم Redis بدل dict)
# ----------------------------------------------------------------------------
_memories: dict[str, ChatMemoryBuffer] = {}


def get_memory(session_id: str) -> ChatMemoryBuffer:
    if session_id not in _memories:
        _memories[session_id] = ChatMemoryBuffer.from_defaults(token_limit=3000)
    return _memories[session_id]


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
APP_PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

app = FastAPI(title="University RAG", description="UniFlow API + dashboard for web & mobile")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False if CORS_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                ips.append(ip)
        except OSError:
            pass
    return ips

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".txt", ".md"}


def _looks_like_garbage(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    if text.lstrip().startswith("%PDF"):
        return True
    sample = text[:2000]
    printable = sum(1 for ch in sample if ch.isprintable() or ch.isspace())
    return (printable / max(len(sample), 1)) < 0.85


def load_file_documents(path: Path) -> list[Document]:
    """استخراج نص نظيف — PyMuPDF للـ PDF بدل قراءة الباينري."""
    if path.suffix.lower() == ".pdf":
        pdf = pymupdf.open(path)
        docs: list[Document] = []
        for i, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={"page_label": str(i + 1)},
                )
            )
        pdf.close()
        return docs
    return SimpleDirectoryReader(input_files=[str(path)]).load_data()


class CourseIn(BaseModel):
    title: Optional[str] = None


class CourseOut(BaseModel):
    course_id: str
    title: str


class StudentUpsert(BaseModel):
    name: Optional[str] = None
    courses: list[str] = Field(default_factory=list, description="مواد الطالب")


class StudentOut(BaseModel):
    student_id: str
    name: str
    courses: list[str]


class ChatRequest(BaseModel):
    # التطبيق الموبايل أحيانًا ما بيبعت student_id → افتراضي s1 للتجربة
    student_id: Optional[str] = "s1"
    session_id: str
    question: str
    course_id: Optional[str] = None


api = APIRouter(prefix="/api", tags=["mobile-api"])


@app.get("/")
async def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/courses", response_model=list[CourseOut])
async def list_courses():
    return db.list_courses()


@app.put("/courses/{course_id}", response_model=CourseOut)
async def upsert_course(course_id: str, body: CourseIn = CourseIn()):
    course_id = course_id.strip()
    if not course_id:
        raise HTTPException(400, "course_id فاضي.")
    return db.upsert_course(course_id, body.title or course_id)


@app.delete("/courses/{course_id}")
async def delete_course(course_id: str):
    if not db.delete_course(course_id):
        raise HTTPException(404, "المادة غير موجودة.")
    return {"status": "deleted", "course_id": course_id}


@app.get("/students", response_model=list[StudentOut])
async def list_students():
    return db.list_students()


@app.get("/students/{student_id}", response_model=StudentOut)
async def get_student(student_id: str):
    student = db.get_student(student_id)
    if not student:
        raise HTTPException(404, "الطالب غير موجود.")
    return student


@app.put("/students/{student_id}", response_model=StudentOut)
async def upsert_student(student_id: str, body: StudentUpsert):
    """أضف طالب جديد أو حدّث اسمه ومواده (يستبدل قائمة المواد كاملة)."""
    student_id = student_id.strip()
    if not student_id:
        raise HTTPException(400, "student_id فاضي.")
    return db.upsert_student(student_id, body.name or student_id, body.courses)


@app.post("/students/{student_id}/courses/{course_id}", response_model=StudentOut)
async def enroll_course(student_id: str, course_id: str):
    """سجّل طالب بمادة إضافية."""
    student_id, course_id = student_id.strip(), course_id.strip()
    if not student_id or not course_id:
        raise HTTPException(400, "student_id أو course_id فاضي.")
    return db.enroll(student_id, course_id)


@app.delete("/students/{student_id}/courses/{course_id}", response_model=StudentOut)
async def unenroll_course(student_id: str, course_id: str):
    student = db.unenroll(student_id, course_id)
    if not student:
        raise HTTPException(404, "الطالب غير موجود.")
    return student


@app.delete("/students/{student_id}")
async def delete_student(student_id: str):
    if not db.delete_student(student_id):
        raise HTTPException(404, "الطالب غير موجود.")
    return {"status": "deleted", "student_id": student_id}


@app.post("/ingest")
async def ingest(
    course_id: str = Form(...),
    lecture_id: str = Form(...),
    file: UploadFile = File(...),
):
    """يُستدعى لما الدكتور/الأدمن يرفع ملف محاضرة لمادة."""
    course_id = course_id.strip()
    lecture_id = lecture_id.strip()
    db.upsert_course(course_id, course_id)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"نوع الملف غير مدعوم. المدعوم: {sorted(ALLOWED_EXT)}")

    tmp_dir = tempfile.mkdtemp()
    try:
        path = Path(tmp_dir) / file.filename
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = load_file_documents(path)
        docs = [d for d in docs if not _looks_like_garbage(d.text or "")]
        if not docs:
            raise HTTPException(
                422,
                "ما قدرت أستخرج نص مقروء من الملف (ممكن يكون صور/سكان أو PDF تالف).",
            )

        for d in docs:
            d.metadata = {
                "course_id": course_id,
                "lecture_id": lecture_id,
                "file_name": file.filename,
                "page": d.metadata.get("page_label", ""),
            }
            d.excluded_embed_metadata_keys = ["course_id", "lecture_id"]
            d.excluded_llm_metadata_keys = ["course_id", "lecture_id"]

        # امسح أي نسخة قديمة لنفس الملف بأي course/lecture (حتى لو تغيّر الكود)
        collection.delete(where={"file_name": file.filename})
        for d in docs:
            index.insert(d)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "status": "indexed",
        "course_id": course_id,
        "lecture_id": lecture_id,
        "pages_indexed": len(docs),
        "preview": (docs[0].text or "")[:180],
    }


@app.delete("/lectures/{course_id}/{lecture_id}")
async def delete_lecture(course_id: str, lecture_id: str):
    collection.delete(
        where={"$and": [{"course_id": course_id}, {"lecture_id": lecture_id}]}
    )
    return {"status": "deleted"}


async def _run_chat(req: ChatRequest) -> dict:
    student_id = (req.student_id or "s1").strip() or "s1"
    enrolled = db.get_student_course_ids(student_id)
    if not enrolled:
        raise HTTPException(403, "الطالب غير مسجل بأي مادة.")

    if req.course_id:
        if req.course_id not in enrolled:
            raise HTTPException(403, "أنت غير مسجل بهاي المادة.")
        allowed = [req.course_id]
    else:
        allowed = enrolled

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="course_id", operator=FilterOperator.IN, value=allowed)
        ]
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=get_memory(f"{student_id}:{req.session_id}"),
        system_prompt=SYSTEM_PROMPT,
        similarity_top_k=8,
        filters=filters,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.05)],
    )

    response = await chat_engine.achat(req.question)

    sources = [
        {
            "course_id": n.metadata.get("course_id"),
            "lecture_id": n.metadata.get("lecture_id"),
            "file_name": n.metadata.get("file_name"),
            "page": n.metadata.get("page"),
            "score": round(n.score, 3) if n.score is not None else None,
            "snippet": (n.get_content() or "")[:160],
        }
        for n in response.source_nodes
    ]
    if not sources:
        return {
            "answer": (
                "ما لقيت مقاطع مرتبطة بسؤالك ضمن موادك المسجّلة "
                f"({', '.join(allowed)}). تأكد إن المحاضرة اترفعت بنفس course_id."
            ),
            "sources": [],
            "allowed_courses": allowed,
        }
    return {"answer": str(response), "sources": sources, "allowed_courses": allowed}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Web dashboard: POST /chat"""
    return await _run_chat(req)


@api.post("/chat")
async def api_chat(req: ChatRequest):
    """Flutter mobile: POST /api/chat"""
    return await _run_chat(req)


def _health_payload(request: Request) -> dict:
    port = request.url.port or APP_PORT
    lan = _lan_ips()
    return {
        "ok": True,
        "chunks": collection.count(),
        "students": len(db.list_students()),
        "courses": len(db.list_courses()),
        "db": str(db.DB_PATH),
        "port": port,
        "web": f"http://127.0.0.1:{port}/",
        "mobile_urls": [f"http://{ip}:{port}/" for ip in lan],
        "api_docs": f"http://127.0.0.1:{port}/docs",
        "chat_endpoints": ["/chat", "/api/chat"],
    }


@app.get("/health")
async def health(request: Request):
    return _health_payload(request)


@api.get("/health")
async def api_health(request: Request):
    return _health_payload(request)


app.include_router(api)
