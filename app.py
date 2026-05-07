from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)


app.secret_key = "chatbot_secret_key_123"   # REQUIRED for session


# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "jobs.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cosine_matrix.csv")
# Define CSV path correctly
COURSE_PATH = os.path.join(BASE_DIR, "data", "courses.csv")

# Load CSV
courses_df = pd.read_csv(COURSE_PATH)

# Clean column names (important)
courses_df.columns = courses_df.columns.str.strip()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", total_jobs=len(df))

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# CSV columns:
# company_name, address, job_title, required_skills, apply_link

# Fill missing values safely
df["company_name"] = df["company_name"].fillna("")
df["address"] = df["address"].fillna("")
df["job_title"] = df["job_title"].fillna("")
df["required_skills"] = df["required_skills"].fillna("")
df["apply_link"] = df["apply_link"].fillna("")

# ---------------- AI MODEL ----------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["required_skills"])
cosine_matrix = cosine_similarity(tfidf_matrix)


# ------------------------------------------------
# SKILL → JOB → COMPANY
# ------------------------------------------------
@app.route("/skill_recommend", methods=["GET", "POST"])
def skill_recommend():
    results = None

    if request.method == "POST":
        skill = request.form.get("skill")

        results = df[
            df["required_skills"].str.contains(skill, case=False, na=False)
        ][["job_title", "company_name", "address", "apply_link"]].drop_duplicates()

    return render_template("skill_recommend.html", results=results)

# ------------------------------------------------
# JOB → COMPANY
# ------------------------------------------------
@app.route("/job_company", methods=["GET", "POST"])
def job_company():
    results = None
    if request.method == "POST":
        job_title = request.form.get("job_title")
        results = df[
            df["job_title"].str.contains(job_title, case=False, na=False)
        ][["company_name", "address", "apply_link"]].drop_duplicates()
    return render_template("job_company.html", results=results)

# ------------------------------------------------
# COSINE MATRIX DOWNLOAD
# ------------------------------------------------
@app.route("/download_cosine")
def download_cosine():
    os.makedirs(MODEL_DIR, exist_ok=True)
    pd.DataFrame(cosine_matrix).to_csv(MODEL_PATH, index=False)
    return send_file(MODEL_PATH, as_attachment=True)

# ------------------------------------------------
# COURSE REFERENCES
# ------------------------------------------------

@app.route("/courses", methods=["GET", "POST"])
def courses():
    results = None

    if request.method == "POST":
        job_role = request.form.get("job_role")

        if job_role:
            results = courses_df[
                courses_df["job_role"].str.contains(job_role, case=False, na=False)
            ]

    return render_template("courses.html", results=results)

# ------------------------------------------------
# CHATBOT
# ------------------------------------------------
'''@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    response = None
    user_message = None

    # 50+ predefined Q&A
    chatbot_responses = {

        # Job Related
        "what is data scientist": "A Data Scientist analyzes data to extract insights using ML and statistics.",
        "what is python developer": "A Python Developer builds backend systems, APIs, and automation tools.",
        "what is full stack developer": "A Full Stack Developer works on both frontend and backend.",
        "highest paying it jobs": "Top paying IT jobs: Data Scientist, AI Engineer, Cloud Architect, DevOps Engineer.",
        "how to get job in it": "Learn skills, build projects, prepare resume, apply on LinkedIn and job portals.",
        "difference between frontend and backend": "Frontend works on UI. Backend handles server and database logic.",
        "what is machine learning": "Machine Learning allows systems to learn from data automatically.",
        "how to become data analyst": "Learn Excel, SQL, Python, Power BI, and build data projects.",
        "best jobs after btech": "Software Developer, Data Scientist, DevOps Engineer, Cybersecurity Analyst.",
        "remote jobs available": "Yes, many companies offer remote roles in development and data fields.",

        # Skill Related
        "best programming language": "Python is best for beginners. JavaScript for web. Java for enterprise.",
        "is python good for job": "Yes, Python has high demand in AI, Data Science, Web Development.",
        "what skills for data science": "Python, Statistics, Machine Learning, SQL, Data Visualization.",
        "what skills for web developer": "HTML, CSS, JavaScript, React, Node.js.",
        "cloud skills required": "AWS, Azure, Docker, Kubernetes, Linux.",
        "devops skills": "CI/CD, Docker, Kubernetes, AWS, Linux.",
        "ai skills": "Python, TensorFlow, Deep Learning, NLP.",
        "sql important": "Yes, SQL is essential for database management.",
        "communication skills important": "Yes, communication is critical for interviews and teamwork.",
        "how to improve coding": "Practice daily on LeetCode, CodeStudio, and build projects.",

        # Course Related
        "best course for python": "You can take Python courses from Udemy, Coursera, or edX.",
        "best data science course": "Machine Learning by Andrew Ng on Coursera is highly recommended.",
        "free coding courses": "Check freeCodeCamp, Coursera free courses, and YouTube.",
        "aws course": "AWS Cloud Practitioner on Coursera or Udemy.",
        "devops course": "Docker and Kubernetes course on Udemy.",
        "ai course": "Deep Learning Specialization by Andrew Ng.",
        "frontend course": "React JS course on Udemy.",
        "backend course": "Node.js and Express course on Udemy.",
        "cybersecurity course": "CEH and CompTIA Security+ courses.",
        "best certification": "AWS, Azure, Google Cloud certifications are valuable.",

        # Resume Related
        "how to build resume": "Keep it 1 page, highlight skills, projects, internships.",
        "resume format": "Use clean format: Summary, Skills, Projects, Education.",
        "resume for fresher": "Focus on projects, internships, certifications.",
        "should i add photo in resume": "No, unless specifically required.",
        "how many pages resume": "1 page for freshers, 1-2 for experienced.",
        "what to include in resume": "Skills, Projects, Experience, Education.",
        "resume mistakes": "Avoid spelling errors, too much text, outdated info.",
        "how to pass ats": "Use keywords from job description.",
        "resume objective example": "Motivated graduate seeking entry-level software role.",
        "resume summary": "Short 2-3 line professional summary at top.",

        # Interview Related
        "interview tips": "Practice coding, revise basics, mock interviews.",
        "hr interview questions": "Tell me about yourself, strengths, weaknesses.",
        "technical interview tips": "Understand fundamentals and practice coding.",
        "how to prepare for coding interview": "Solve DSA problems daily.",
        "tell me about yourself": "Start with education, skills, projects.",
        "strength answer": "Mention technical strength with example.",
        "weakness answer": "Mention improvement area and how you're working on it.",
        "why should we hire you": "Show your skills match job requirements.",
        "what is your goal": "To grow technically and contribute to company.",
        "salary expectation answer": "Open to discussion based on role."
    }

    if request.method == "POST":
        user_message = request.form.get("message").lower().strip()

        response = chatbot_responses.get(
            user_message,
            "Sorry, I don't understand. Please ask about jobs, skills, courses, resume, or interviews."
        )

    return render_template("chatbot.html",
                           response=response,
                           user_message=user_message)'''

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_message = request.form.get("message")
        bot_response = ""

        if user_message:
            msg = user_message.lower().strip()

            # ---------------- GREETING ----------------
            if any(word in msg for word in ["hi", "hello", "hey", "hii"]):
                bot_response = "Hello 👋 Welcome to AI Job Portal! How can I help you?"

            elif "how are you" in msg:
                bot_response = "I'm doing great 😊 Ready to help you with career guidance!"

            elif any(word in msg for word in ["thank", "thanks"]):
                bot_response = "You're welcome 😊 Happy to help!"

            elif any(word in msg for word in ["bye", "goodbye"]):
                bot_response = "Good luck with your career 🚀 See you soon!"

            # ---------------- JOB RELATED ----------------
            elif "data scientist" in msg:
                bot_response = "A Data Scientist analyzes data using ML and statistics to generate insights."

            elif "python developer" in msg:
                bot_response = "A Python Developer builds backend systems, APIs, automation tools and web apps."

            elif "full stack" in msg:
                bot_response = "A Full Stack Developer works on both frontend and backend technologies."

            elif "highest paying" in msg or "high salary" in msg:
                bot_response = "Top high-paying IT jobs: Data Scientist, AI Engineer, Cloud Architect, DevOps Engineer."

            elif "how to get job" in msg or "get job in it" in msg:
                bot_response = "Learn in-demand skills, build projects, prepare resume and apply through job portals."

            elif "frontend and backend" in msg:
                bot_response = "Frontend handles UI. Backend manages server, database, and business logic."

            elif "machine learning" in msg:
                bot_response = "Machine Learning allows systems to learn patterns from data automatically."

            elif "data analyst" in msg:
                bot_response = "Learn Excel, SQL, Python, Power BI and build real-world data projects."

            elif "remote job" in msg:
                bot_response = "Yes, many IT companies offer remote roles in development and data fields."

            # ---------------- SKILLS ----------------
            elif "best programming language" in msg:
                bot_response = "Python is great for beginners. JavaScript for web. Java for enterprise apps."

            elif "python good" in msg:
                bot_response = "Yes, Python is highly in demand for AI, Web, Automation, and Data Science."

            elif "data science skills" in msg:
                bot_response = "Required skills: Python, Statistics, ML, SQL, Data Visualization."

            elif "web developer skills" in msg:
                bot_response = "HTML, CSS, JavaScript, React, Node.js are essential."

            elif "cloud skills" in msg:
                bot_response = "AWS, Azure, Docker, Kubernetes, Linux are important cloud skills."

            elif "devops" in msg:
                bot_response = "DevOps requires CI/CD, Docker, Kubernetes, AWS and Linux knowledge."

            elif "ai skills" in msg:
                bot_response = "AI skills include Python, TensorFlow, Deep Learning and NLP."

            elif "sql important" in msg:
                bot_response = "Yes, SQL is very important for managing databases."

            elif "improve coding" in msg:
                bot_response = "Practice daily on LeetCode, HackerRank and build projects."

            # ---------------- COURSES ----------------
            elif "python course" in msg:
                bot_response = "Best Python courses are available on Udemy, Coursera and edX."

            elif "data science course" in msg:
                bot_response = "Machine Learning by Andrew Ng on Coursera is highly recommended."

            elif "free course" in msg:
                bot_response = "Check freeCodeCamp, Coursera free courses and YouTube tutorials."

            elif "aws course" in msg:
                bot_response = "AWS Cloud Practitioner course on Coursera or Udemy is good."

            elif "devops course" in msg:
                bot_response = "Docker and Kubernetes course on Udemy is recommended."

            elif "frontend course" in msg:
                bot_response = "React JS course on Udemy or Coursera is good."

            elif "cybersecurity" in msg:
                bot_response = "CEH and CompTIA Security+ certifications are popular."

            elif "certification" in msg:
                bot_response = "AWS, Azure, Google Cloud certifications are valuable."

            # ---------------- RESUME ----------------
            elif "build resume" in msg or "create resume" in msg:
                bot_response = "Keep it 1 page, highlight skills, projects and internships."

            elif "resume format" in msg:
                bot_response = "Use format: Summary, Skills, Projects, Experience, Education."

            elif "resume for fresher" in msg:
                bot_response = "Focus on academic projects, internships and certifications."

            elif "photo in resume" in msg:
                bot_response = "No, adding photo is not required unless specified."

            elif "resume mistakes" in msg:
                bot_response = "Avoid spelling errors, too much text and irrelevant details."

            elif "ats" in msg:
                bot_response = "Use keywords from job description to pass ATS filters."

            elif "resume objective" in msg:
                bot_response = "Write a short career goal aligned with the job role."

            # ---------------- INTERVIEW ----------------
            elif "interview tips" in msg:
                bot_response = "Revise fundamentals, practice coding and attend mock interviews."

            elif "hr interview" in msg:
                bot_response = "Prepare answers for strengths, weaknesses and career goals."

            elif "coding interview" in msg:
                bot_response = "Solve DSA problems daily and understand time complexity."

            elif "tell me about yourself" in msg:
                bot_response = "Start with education, skills, projects and career goals."

            elif "why should we hire you" in msg:
                bot_response = "Explain how your skills match the job requirements."

            elif "salary expectation" in msg:
                bot_response = "Say you are open to discussion based on role and market standards."

            else:
                bot_response = "I can help with jobs, skills, resume, courses and interview guidance 😊"

            # Save chat
            chat_history = session["chat_history"]
            chat_history.append({"user": user_message, "bot": bot_response})
            session["chat_history"] = chat_history
            session.modified = True

    return render_template("chatbot.html",
                           chat_history=session["chat_history"])


@app.route("/get_response", methods=["POST"])
def get_response():
    msg = request.json.get("message", "").lower()

    if "python" in msg:
        reply = "Python is widely used in Data Science, AI, and ML."
    elif "job" in msg:
        reply = "Try roles like Data Scientist, ML Engineer, or Software Developer."
    elif "resume" in msg:
        reply = "Focus on skills, internships, and projects."
    elif "course" in msg:
        reply = "Python for Data Science and Machine Learning A-Z are good choices."
    elif "hi" in msg or "hello" in msg:
        reply = "Hello 👋 I am your AI Recruitment Assistant."
    else:
        reply = "Ask me about jobs, skills, resumes, or courses."

    return jsonify({"reply": reply})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
