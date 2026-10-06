import os
import re
import json
import uuid
import sqlite3
import tempfile
from datetime import datetime, date, timedelta

import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide",
)

DB_FILE = "student_assistant.db"
CHROMA_DIR = "./chroma_db"

DEFAULT_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Change if required
OLLAMA_MODEL = st.session_state.get(
    "ollama_model",
    DEFAULT_MODEL
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return sqlite3.connect(
        DB_FILE,
        check_same_thread=False
    )


def init_db():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            due_datetime TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            notified INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            document_type TEXT,
            uploaded_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            topic TEXT,
            difficulty TEXT,
            next_review TEXT,
            mastered INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            score INTEGER,
            total INTEGER,
            taken_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


@st.cache_resource
def load_whisper():

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


@st.cache_resource
def load_chroma():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name="student_documents",
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


embedding_model = load_embedding_model()
collection = load_chroma()


# ============================================================
# OLLAMA
# ============================================================

def ask_ai(
    prompt,
    context="",
    system=None
):

    if system is None:

        system = """
You are a personal AI assistant for a university student.

Your job is to help the student learn effectively.

You can:
- Explain difficult concepts.
- Analyze syllabus and previous-year papers.
- Find important topics.
- Create study plans.
- Create flashcards.
- Create quizzes.
- Answer academic doubts.
- Summarize notes.

When document context is supplied, use it as the primary source.

Do not invent information from uploaded documents.
If information is unavailable, clearly say so.
"""

    if context:

        prompt = f"""
Relevant student documents:

-----------------------------
{context}
-----------------------------

Student request:

{prompt}
"""

    try:

        result = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return result["message"]["content"]

    except Exception as e:

        return f"""
⚠️ Ollama connection failed.

Make sure Ollama is running and the model
`{OLLAMA_MODEL}` is installed.

Run:

ollama pull {OLLAMA_MODEL}

Error:
{e}
"""


# ============================================================
# EMBEDDINGS / RAG
# ============================================================

def get_embedding(text):

    vector = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def split_text(
    text,
    chunk_size=900,
    overlap=150
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def extract_pdf(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    pages = []

    for page_no, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append(
                f"[Page {page_no}]\n{text}"
            )

    return "\n\n".join(pages)


def add_pdf_to_rag(
    uploaded_file,
    document_type
):

    text = extract_pdf(
        uploaded_file
    )

    if not text.strip():
        return 0

    chunks = split_text(text)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    doc_id = str(
        uuid.uuid4()
    )

    for i, chunk in enumerate(chunks):

        ids.append(
            f"{doc_id}_{i}"
        )

        documents.append(chunk)

        embeddings.append(
            get_embedding(chunk)
        )

        metadatas.append({
            "filename": uploaded_file.name,
            "document_type": document_type,
            "chunk": i
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    conn = get_db()

    conn.execute(
        """
        INSERT OR REPLACE INTO documents
        (filename, document_type, uploaded_at)
        VALUES (?, ?, ?)
        """,
        (
            uploaded_file.name,
            document_type,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()

    return len(chunks)


def search_rag(
    query,
    n=6,
    document_type=None
):

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(
        query
    )

    kwargs = {
        "query_embeddings": [
            query_embedding
        ],
        "n_results": min(
            n,
            collection.count()
        )
    }

    if document_type:

        kwargs["where"] = {
            "document_type": document_type
        }

    try:

        result = collection.query(
            **kwargs
        )

    except Exception:

        return []

    docs = result.get(
        "documents",
        [[]]
    )[0]

    metadata = result.get(
        "metadatas",
        [[]]
    )[0]

    output = []

    for doc, meta in zip(
        docs,
        metadata
    ):

        output.append({
            "text": doc,
            "filename": meta.get(
                "filename",
                "Unknown"
            ),
            "type": meta.get(
                "document_type",
                ""
            )
        })

    return output


def make_context(results):

    return "\n\n".join(
        f"Source: {x['filename']}\n{x['text']}"
        for x in results
    )


# ============================================================
# DOCUMENT ANALYSIS
# ============================================================

def analyze_important_topics():

    syllabus = search_rag(
        "syllabus units subjects topics",
        n=12,
        document_type="syllabus"
    )

    pyqs = search_rag(
        "previous year questions important repeated questions",
        n=20,
        document_type="pyq"
    )

    if not syllabus:
        return "Please upload your syllabus PDF first."

    if not pyqs:
        return "Please upload previous-year question papers."

    context = (
        "SYLLABUS:\n"
        + make_context(syllabus)
        + "\n\nPREVIOUS YEAR PAPERS:\n"
        + make_context(pyqs)
    )

    prompt = """
Analyze the syllabus and previous-year question papers.

Identify important exam topics.

For every topic provide:

1. Topic name
2. Unit
3. Number/frequency of appearances in PYQs
4. Priority: Very High / High / Medium / Low
5. Why it is important
6. Recommended preparation strategy

Look for repeated questions and concepts even when
the wording is different.

Do not invent topics that aren't supported by the
provided documents.

Return a clean Markdown table followed by
recommendations.
"""

    return ask_ai(
        prompt,
        context
    )


# ============================================================
# FLASHCARDS
# ============================================================

def generate_flashcards(topic):

    results = search_rag(
        topic,
        n=8
    )

    if not results:

        return []

    context = make_context(
        results
    )

    prompt = f"""
Create 10 useful flashcards about:

{topic}

Use ONLY the supplied study material.

Return JSON exactly like:

[
  {{
    "question": "...",
    "answer": "...",
    "topic": "{topic}",
    "difficulty": "easy"
  }}
]

Difficulty must be easy, medium, or hard.

Do not add Markdown.
"""

    response = ask_ai(
        prompt,
        context
    )

    try:

        match = re.search(
            r"\[.*\]",
            response,
            re.DOTALL
        )

        if not match:
            return []

        return json.loads(
            match.group()
        )

    except Exception:

        return []


def save_flashcards(cards):

    conn = get_db()

    for card in cards:

        conn.execute(
            """
            INSERT INTO flashcards
            (question, answer, topic, difficulty, next_review)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                card.get(
                    "question",
                    ""
                ),
                card.get(
                    "answer",
                    ""
                ),
                card.get(
                    "topic",
                    ""
                ),
                card.get(
                    "difficulty",
                    "medium"
                ),
                date.today().isoformat()
            )
        )

    conn.commit()
    conn.close()


def get_flashcards():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT id, question, answer, topic,
               difficulty, mastered
        FROM flashcards
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return rows


def mark_flashcard(
    card_id,
    mastered
):

    conn = get_db()

    next_review = (
        date.today() + timedelta(days=7)
    ).isoformat() if mastered else (
        date.today()
    ).isoformat()

    conn.execute(
        """
        UPDATE flashcards
        SET mastered = ?,
            next_review = ?
        WHERE id = ?
        """,
        (
            int(mastered),
            next_review,
            card_id
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# QUIZ
# ============================================================

def generate_quiz(topic):

    results = search_rag(
        topic,
        n=10
    )

    context = make_context(
        results
    )

    prompt = f"""
Create a 10-question university-level quiz
about {topic}.

Use the supplied study material.

Return ONLY valid JSON:

[
  {{
    "question": "...",
    "options": [
      "A. ...",
      "B. ...",
      "C. ...",
      "D. ..."
    ],
    "answer": "A",
    "explanation": "..."
  }}
]

Each question must have exactly four options.
"""

    response = ask_ai(
        prompt,
        context
    )

    try:

        match = re.search(
            r"\[.*\]",
            response,
            re.DOTALL
        )

        if not match:
            return []

        return json.loads(
            match.group()
        )

    except Exception:

        return []


# ============================================================
# STUDY PLAN
# ============================================================

def generate_study_plan(
    timetable,
    exam_dates,
    hours
):

    syllabus = search_rag(
        "subjects units syllabus topics",
        n=20,
        document_type="syllabus"
    )

    pyqs = search_rag(
        "important repeated previous questions",
        n=20,
        document_type="pyq"
    )

    context = make_context(
        syllabus + pyqs
    )

    prompt = f"""
Create a personalized semester exam preparation plan.

Student semester timetable:
{timetable}

Exam dates:
{exam_dates}

Available study time:
{hours} hours/day

Use the uploaded syllabus and previous-year papers.

Requirements:

- Give priority to upcoming exams.
- Give extra time to difficult/high-frequency topics.
- Include revision.
- Include PYQ practice.
- Include mock tests.
- Include flashcard sessions.
- Include breaks.
- Avoid scheduling study during classes.
- Give a day-by-day plan.
- Make it realistic.

Return Markdown.
"""

    return ask_ai(
        prompt,
        context
    )


# ============================================================
# TASKS
# ============================================================

def add_task(
    title,
    description,
    due
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO tasks
        (title, description, due_datetime)
        VALUES (?, ?, ?)
        """,
        (
            title,
            description,
            due
        )
    )

    conn.commit()
    conn.close()


def get_tasks():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT id, title, description,
               due_datetime, completed
        FROM tasks
        ORDER BY due_datetime
        """
    ).fetchall()

    conn.close()

    return rows


def complete_task(task_id):

    conn = get_db()

    conn.execute(
        """
        UPDATE tasks
        SET completed = 1
        WHERE id = ?
        """,
        (task_id,)
    )

    conn.commit()
    conn.close()


def due_tasks():

    now = datetime.now()

    tasks = get_tasks()

    result = []

    for task in tasks:

        if task[4]:
            continue

        try:

            task_time = datetime.fromisoformat(
                task[3]
            )

            if task_time <= now:

                result.append(task)

        except Exception:

            pass

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎓 Student AI Assistant"
)

st.sidebar.caption(
    "Your personal AI study companion"
)

model = st.sidebar.text_input(
    "Ollama model",
    value=OLLAMA_MODEL
)

OLLAMA_MODEL = model
st.session_state.ollama_model = model

page = st.sidebar.radio(
    "Menu",
    [
        "🏠 Dashboard",
        "🤖 AI Doubts",
        "📚 Documents",
        "🔥 Important Topics",
        "🧠 Flashcards",
        "📝 Quiz",
        "📅 Study Plan",
        "⏰ To-Do & Reminders",
        "🎙️ Voice Assistant"
    ]
)


# ============================================================
# GLOBAL REMINDERS
# ============================================================

due = due_tasks()

if due:

    st.warning(
        f"🔔 You have {len(due)} task(s) that are due now!"
    )

    for task in due:

        st.error(
            f"⏰ **{task[1]}**\n\n"
            f"{task[2] or ''}\n\n"
            f"Scheduled: {task[3]}"
        )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🎓 Student AI Dashboard"
    )

    st.write(
        f"Today: **{datetime.now().strftime('%A, %d %B %Y')}**"
    )

    tasks = get_tasks()

    pending = [
        x for x in tasks
        if not x[4]
    ]

    cards = get_flashcards()

    mastered = [
        x for x in cards
        if x[5]
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📋 Pending Tasks",
        len(pending)
    )

    c2.metric(
        "🧠 Flashcards",
        len(cards)
    )

    c3.metric(
        "✅ Mastered",
        len(mastered)
    )

    c4.metric(
        "📚 Document Chunks",
        collection.count()
    )

    st.divider()

    st.subheader(
        "⏰ Today's Tasks"
    )

    for task in pending[:5]:

        st.write(
            f"**{task[1]}** — {task[3]}"
        )

    st.divider()

    st.subheader(
        "🚀 What do you want to do?"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            "📚 Upload syllabus + PYQs\n\n"
            "Find the topics most likely to matter."
        )

    with c2:

        st.info(
            "🧠 Practice flashcards\n\n"
            "Use active recall to remember concepts."
        )

    with c3:

        st.info(
            "📝 Take a quiz\n\n"
            "Test yourself and find weak topics."
        )


# ============================================================
# AI DOUBTS
# ============================================================

elif page == "🤖 AI Doubts":

    st.title(
        "🤖 Ask Your AI Study Assistant"
    )

    st.caption(
        "Ask doubts about your subjects or uploaded material."
    )

    if "chat" not in st.session_state:

        st.session_state.chat = []

    for message in st.session_state.chat:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    prompt = st.chat_input(
        "Ask your doubt..."
    )

    if prompt:

        st.session_state.chat.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        results = search_rag(
            prompt,
            n=6
        )

        context = make_context(
            results
        )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_ai(
                    prompt,
                    context
                )

                st.markdown(
                    answer
                )

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer
        })


# ============================================================
# DOCUMENTS
# ============================================================

elif page == "📚 Documents":

    st.title(
        "📚 Study Materials"
    )

    st.write(
        "Upload your syllabus, PYQs, notes and timetable."
    )

    document_type = st.selectbox(
        "What are you uploading?",
        [
            "syllabus",
            "pyq",
            "notes",
            "timetable"
        ]
    )

    files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:

        for file in files:

            with st.spinner(
                f"Processing {file.name}..."
            ):

                try:

                    count = add_pdf_to_rag(
                        file,
                        document_type
                    )

                    st.success(
                        f"✅ {file.name}: "
                        f"{count} chunks indexed."
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )

    st.divider()

    conn = get_db()

    documents = conn.execute(
        """
        SELECT filename,
               document_type,
               uploaded_at
        FROM documents
        ORDER BY uploaded_at DESC
        """
    ).fetchall()

    conn.close()

    for doc in documents:

        st.write(
            f"📄 **{doc[0]}** — "
            f"{doc[1]} — {doc[2]}"
        )


# ============================================================
# IMPORTANT TOPICS
# ============================================================

elif page == "🔥 Important Topics":

    st.title(
        "🔥 Important Exam Topics"
    )

    st.write(
        "I'll compare your syllabus with previous-year papers."
    )

    if st.button(
        "🔍 Analyze Important Topics",
        type="primary"
    ):

        with st.spinner(
            "Analyzing syllabus and PYQs..."
        ):

            result = analyze_important_topics()

        st.markdown(
            result
        )


# ============================================================
# FLASHCARDS
# ============================================================

elif page == "🧠 Flashcards":

    st.title(
        "🧠 Active Recall Flashcards"
    )

    topic = st.text_input(
        "Topic",
        placeholder="e.g. Operating System scheduling"
    )

    if st.button(
        "✨ Generate Flashcards",
        type="primary"
    ):

        if not topic:

            st.warning(
                "Enter a topic first."
            )

        else:

            with st.spinner(
                "Creating flashcards..."
            ):

                cards = generate_flashcards(
                    topic
                )

            if cards:

                save_flashcards(
                    cards
                )

                st.success(
                    f"Created {len(cards)} flashcards!"
                )

                st.session_state.flashcard_data = cards

            else:

                st.error(
                    "Could not generate flashcards."
                )

    if "flashcard_data" in st.session_state:

        st.divider()

        st.subheader(
            "📖 New Flashcards"
        )

        for card in st.session_state.flashcard_data:

            with st.expander(
                card["question"]
            ):

                st.write(
                    "### Answer"
                )

                st.write(
                    card["answer"]
                )

                st.caption(
                    f"Difficulty: {card.get('difficulty', 'medium')}"
                )

    st.divider()

    st.subheader(
        "📚 Saved Flashcards"
    )

    saved = get_flashcards()

    for card in saved[:20]:

        card_id, question, answer, topic, difficulty, mastered = card

        with st.expander(
            question
        ):

            st.write(answer)

            if mastered:

                st.success(
                    "✅ Mastered"
                )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    if st.button(
                        "✅ I know this",
                        key=f"know_{card_id}"
                    ):

                        mark_flashcard(
                            card_id,
                            True
                        )

                        st.rerun()

                with c2:

                    if st.button(
                        "🔄 Need practice",
                        key=f"practice_{card_id}"
                    ):

                        mark_flashcard(
                            card_id,
                            False
                        )

                        st.rerun()


# ============================================================
# QUIZ
# ============================================================

elif page == "📝 Quiz":

    st.title(
        "📝 AI Quiz"
    )

    topic = st.text_input(
        "Quiz topic",
        placeholder="e.g. Computer Networks"
    )

    if st.button(
        "🚀 Generate Quiz",
        type="primary"
    ):

        with st.spinner(
            "Creating quiz..."
        ):

            quiz = generate_quiz(
                topic
            )

        if quiz:

            st.session_state.quiz = quiz

        else:

            st.error(
                "Could not generate quiz."
            )

    if "quiz" in st.session_state:

        quiz = st.session_state.quiz

        st.divider()

        answers = {}

        for i, q in enumerate(quiz):

            st.subheader(
                f"Question {i + 1}"
            )

            st.write(
                q["question"]
            )

            answers[i] = st.radio(
                "Choose an answer",
                q["options"],
                key=f"quiz_{i}"
            )

        if st.button(
            "📊 Submit Quiz"
        ):

            score = 0

            for i, q in enumerate(quiz):

                selected = answers[i]

                correct = q["answer"]

                if selected.startswith(
                    correct
                ):

                    score += 1

            st.success(
                f"🎉 Score: {score}/{len(quiz)}"
            )

            conn = get_db()

            conn.execute(
                """
                INSERT INTO quiz_scores
                (topic, score, total, taken_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    topic,
                    score,
                    len(quiz),
                    datetime.now().isoformat()
                )
            )

            conn.commit()
            conn.close()

            st.subheader(
                "📖 Explanations"
            )

            for i, q in enumerate(quiz):

                st.write(
                    f"**Q{i+1}:** {q['explanation']}"
                )


# ============================================================
# STUDY PLAN
# ============================================================

elif page == "📅 Study Plan":

    st.title(
        "📅 Personalized Exam Study Plan"
    )

    st.write(
        "Upload your semester timetable first, "
        "then enter your exam information."
    )

    exam_dates = st.text_area(
        "Exam dates",
        placeholder="""
        Mathematics: 2026-11-20
        DBMS: 2026-11-23
        Operating Systems: 2026-11-26
        """
    )

    timetable = st.text_area(
        "Semester timetable / class schedule",
        placeholder="""
        Monday:
        9-10 Data Structures
        10-11 DBMS

        Tuesday:
        9-10 Mathematics
        """
    )

    hours = st.slider(
        "Available study hours per day",
        1,
        12,
        3
    )

    if st.button(
        "🧠 Generate My Study Plan",
        type="primary"
    ):

        if not exam_dates:

            st.warning(
                "Enter your exam dates."
            )

        else:

            with st.spinner(
                "Building your personalized plan..."
            ):

                plan = generate_study_plan(
                    timetable,
                    exam_dates,
                    hours
                )

            st.markdown(
                plan
            )


# ============================================================
# TODO
# ============================================================

elif page == "⏰ To-Do & Reminders":

    st.title(
        "⏰ To-Do List & Reminders"
    )

    st.write(
        "Add study tasks with a specific time."
    )

    title = st.text_input(
        "Task",
        placeholder="Study DBMS normalization"
    )

    description = st.text_area(
        "Description",
        placeholder="Revise 3NF and BCNF"
    )

    due_date = st.date_input(
        "Date",
        date.today()
    )

    due_time = st.time_input(
        "Reminder time"
    )

    if st.button(
        "➕ Add Reminder",
        type="primary"
    ):

        due = datetime.combine(
            due_date,
            due_time
        )

        add_task(
            title,
            description,
            due.isoformat()
        )

        st.success(
            f"Reminder added for {due.strftime('%d %b %Y at %I:%M %p')}"
        )

        st.rerun()

    st.divider()

    st.subheader(
        "📋 My To-Do List"
    )

    tasks = get_tasks()

    for task in tasks:

        task_id, title, description, due, completed = task

        col1, col2, col3 = st.columns(
            [5, 2, 1]
        )

        with col1:

            if completed:

                st.markdown(
                    f"~~{title}~~"
                )

            else:

                st.write(
                    f"📌 **{title}**"
                )

                if description:
                    st.caption(
                        description
                    )

        with col2:

            st.write(
                due
            )

        with col3:

            if not completed:

                if st.button(
                    "✓",
                    key=f"done_{task_id}"
                ):

                    complete_task(
                        task_id
                    )

                    st.rerun()


# ============================================================
# VOICE
# ============================================================

elif page == "🎙️ Voice Assistant":

    st.title(
        "🎙️ Voice Study Assistant"
    )

    audio = st.file_uploader(
        "Upload lecture audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "webm"
        ]
    )

    if audio:

        st.audio(
            audio
        )

        if st.button(
            "🎙️ Transcribe"
        ):

            with st.spinner(
                "Transcribing lecture..."
            ):

                model = load_whisper()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".audio"
                ) as temp:

                    temp.write(
                        audio.read()
                    )

                    path = temp.name

                try:

                    segments, info = model.transcribe(
                        path
                    )

                    transcript = " ".join(
                        segment.text
                        for segment in segments
                    )

                    st.subheader(
                        "📝 Transcript"
                    )

                    st.text_area(
                        "Lecture transcript",
                        transcript,
                        height=300
                    )

                    if st.button(
                        "🧠 Summarize Lecture"
                    ):

                        answer = ask_ai(
                            f"""
                            Summarize this lecture.

                            Also provide:
                            - Important concepts
                            - Key definitions
                            - Exam-important points
                            - 5 flashcard questions
                            - 5 quiz questions

                            Lecture:

                            {transcript}
                            """
                        )

                        st.markdown(
                            answer
                        )

                finally:

                    if os.path.exists(path):
                        os.remove(path)