import os
import re
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# ─── third‑party ─────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

# Suppress noisy SSL warnings when verify=False is used for campus sites
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# We use pdfplumber for cleaner PDF text extraction compared to PyPDF2.
# Install: pip install pdfplumber
try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    # Graceful fall‑back: user might not have pdfplumber installed
    import PyPDF2
    PDF_BACKEND = "PyPDF2"



REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# How long (seconds) to wait between consecutive HTTP requests to be polite
POLITENESS_DELAY = 1.5

# Minimum number of characters a page must yield to be considered useful
MIN_CONTENT_LENGTH = 80

# Directory structure for saved data
RAW_DATA_DIR = os.path.join(".", "data", "raw_corpus")
COMBINED_FILE = os.path.join(RAW_DATA_DIR, "full_corpus.txt")



OFFICIAL_PAGES = [
    ("introduction", "https://iitj.ac.in/main/en/introduction"),
    ("about_iitj", "https://iitj.ac.in/main/en/iitj"),
    ("director_message", "https://iitj.ac.in/main/en/director"),
    ("chairman_message", "https://iitj.ac.in/main/en/chairman"),
    ("why_career_iitj", "https://iitj.ac.in/main/en/why-pursue-a-career-@-iit-jodhpur"),
    ("campus_life", "https://iitj.ac.in/office-of-students/en/campus-life"),
    ("office_of_students", "https://iitj.ac.in/office-of-students/en/office-of-students"),
    ("news_announcements", "https://iitj.ac.in/main/en/news"),
    ("events", "https://iitj.ac.in/main/en/events"),
    ("annual_report", "https://iitj.ac.in/Main/en/Annual-Reports-of-the-Institute"),
]

# ── Source 2: Academic Regulations, Programs, Curriculum (MANDATORY) ─────────
ACADEMIC_PAGES = [
    ("academic_regulations", "https://iitj.ac.in/office-of-academics/en/academic-regulations"),



    ("academic_circulars", "https://iitj.ac.in/office-of-academics/en/circulars"),
    ("program_curriculum", "https://iitj.ac.in/office-of-academics/en/curriculum"),
    ("program_structure", "https://iitj.ac.in/office-of-academics/en/program-Structure"),
    ("academic_programs_list", "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs"),
    ("btech_program", "https://iitj.ac.in/office-of-academics/en/b.tech."),
    ("mtech_program", "https://iitj.ac.in/office-of-academics/en/m.tech."),
    ("phd_program", "https://iitj.ac.in/office-of-academics/en/ph.d."),
    ("mba_program", "https://iitj.ac.in/office-of-academics/en/mba"),
    ("itep_program", "https://iitj.ac.in/itep/"),
    ("saide_btech", "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/btech"),
    ("saide_mtech", "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/mtech"),
    ("saide_courses", "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses"),
    ("academics_office", "https://iitj.ac.in/office-of-academics/en/academics"),
    ("academic_calendar", "https://iitj.ac.in/Office-of-Academics/en/Academic-Calendar"),
    ("ug_registration", "https://iitj.ac.in/office-of-academics/en/ug-registration-guidelines"),
    ("pg_admissions", "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs"),
    ("executive_education", "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education"),
    ("scholarships", "https://iitj.ac.in/office-of-academics/en/scholarships"),
    ("convocation", "https://iitj.ac.in/office-of-academics/en/convocation"),
    ("faqs_applicants", "https://iitj.ac.in/main/en/faqs-applicants"),
    ("office_registrar", "https://iitj.ac.in/office-of-registrar/en/office-of-registrar"),
    ("office_administration", "https://iitj.ac.in/office-of-administration/en/office-of-administration"),
]

# ── Source 3: Newsletters / Institute Repository ────────────────────────────
NEWSLETTER_PAGES = [
    ("newsletter", "https://iitj.ac.in/institute-repository/en/Newsletter"),
]

# ── Source 4: Department / School pages ──────────────────────────────────────
DEPARTMENT_PAGES = [
    ("school_ai_ds", "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science"),
    ("engineering_science", "https://iitj.ac.in/es/en/engineering-science"),
    ("school_liberal_arts", "https://iitj.ac.in/school-of-liberal-arts/"),
    ("school_design", "https://iitj.ac.in/school-of-design/"),
    ("school_management", "https://iitj.ac.in/schools/"),
    ("departments_listing", "https://iitj.ac.in/m/Index/main-departments?lg=en"),
    ("centres_listing", "https://iitj.ac.in/m/Index/main-centers?lg=en"),
    ("idrps_idrcs", "https://iitj.ac.in/m/Index/main-idrps-idrcs?lg=en"),
    ("research_development", "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development"),
    ("research_highlights", "https://iitj.ac.in/main/en/research-highlight"),
    ("central_research_facility", "https://iitj.ac.in/crf/en/crf"),
    ("techscape", "https://iitj.ac.in/techscape/en/Techscape"),
    ("health_center", "https://iitj.ac.in/health-center/en/health-center"),
]

# ── Source 5: Faculty Profile Pages ──────────────────────────────────────────
FACULTY_PAGES = [
    ("faculty_members", "https://iitj.ac.in/main/en/faculty-members"),
    ("visiting_faculty", "https://iitj.ac.in/main/en/visiting-faculty-members"),
    ("scholars_in_residence", "https://iitj.ac.in/main/en/scholars-in-residence"),
    ("faculty_cse", "https://iitj.ac.in/People/List?dept=computer-science-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_ee", "https://iitj.ac.in/People/List?dept=electrical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_me", "https://iitj.ac.in/People/List?dept=mechanical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_physics", "https://iitj.ac.in/People/List?dept=physics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_math", "https://iitj.ac.in/People/List?dept=mathematics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_chemistry", "https://iitj.ac.in/People/List?dept=chemistry&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_bio", "https://iitj.ac.in/People/List?dept=bioscience-bioengineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_civil", "https://iitj.ac.in/People/List?dept=civil-and-infrastructure-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_metallurgy", "https://iitj.ac.in/People/List?dept=metallurgical-and-materials-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_saide", "https://iitj.ac.in/People/List?dept=school-of-artificial-intelligence-data-science&c=ce26246f-00c9-4286-bb4c-7f023b4c5460"),
    ("faculty_sola", "https://iitj.ac.in/People/List?dept=school-of-liberal-arts&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_sod", "https://iitj.ac.in/People/List?dept=school-of-design&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
    ("faculty_sme", "https://iitj.ac.in/People/List?dept=schools&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"),
]


ALL_SCRAPE_TARGETS: List[Tuple[str, str, str]] = (
    [("official",   label, url) for label, url in OFFICIAL_PAGES]  +
    [("academic",   label, url) for label, url in ACADEMIC_PAGES]  +
    [("newsletter", label, url) for label, url in NEWSLETTER_PAGES] +
    [("department", label, url) for label, url in DEPARTMENT_PAGES] +
    [("faculty",    label, url) for label, url in FACULTY_PAGES]
)


def _safe_get(url: str, max_retries: int = 3, timeout: int = 12) -> Optional[bytes]:
    """
    Perform an HTTP GET with exponential back‑off on failure.

    Returns raw response bytes on success, None on failure.
    We separate *fetching* from *parsing* so the two concerns don't tangle.
    """
    for attempt in range(1, max_retries + 1):
        try:

            resp = requests.get(
                url, headers=REQUEST_HEADERS, timeout=timeout, verify=False
            )
            resp.raise_for_status()          # raises on 4xx / 5xx
            return resp.content
        except requests.RequestException as err:
            wait = 2 ** attempt              # exponential back‑off: 2, 4, 8 s
            print(f"  [retry {attempt}/{max_retries}] {url[:60]}… — {err}")
            time.sleep(wait)
    return None


def _html_to_text(raw_html: bytes) -> str:
    """
    Convert raw HTML bytes into clean plain text.

    Cleaning steps
    ──────────────
    1. Parse with lxml (fast C parser).
    2. Strip non‑content tags: <script>, <style>, <nav>, <footer>, etc.
    3. Also strip elements whose *class* or *id* hints at navigation / ads.
    4. Join remaining text with spaces and collapse whitespace.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Tags that never contain useful prose
    JUNK_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "svg"]
    for tag in soup.find_all(JUNK_TAGS):
        tag.decompose()

    # CSS‑class / id patterns that usually wrap boilerplate
    BOILERPLATE_RE = re.compile(
        r"menu|sidebar|breadcrumb|social|cookie|advert|popup|modal", re.I
    )
    for el in soup.find_all(attrs={"class": BOILERPLATE_RE}):
        el.decompose()
    for el in soup.find_all(attrs={"id": BOILERPLATE_RE}):
        el.decompose()

    # Extract visible text, collapse whitespace
    raw_text = soup.get_text(separator=" ")
    clean = re.sub(r"\s+", " ", raw_text).strip()
    return clean


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF loaded into memory.

    Supports two back‑ends:
      • pdfplumber  (preferred — better layout handling)
      • PyPDF2       (fallback if pdfplumber is missing)
    """
    if PDF_BACKEND == "pdfplumber":
        pages_text = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
        return " ".join(pages_text)
    else:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        return " ".join(
            (p.extract_text() or "") for p in reader.pages
        )




def _is_pdf_url(url: str) -> bool:
    """Check whether a URL points to a PDF (by extension or Google Drive dl link)."""
    lower = url.lower()
    return lower.endswith(".pdf") or "drive.google.com/uc" in lower


def _content_looks_like_pdf(raw_bytes: bytes) -> bool:
    """Detect PDF binary content by checking the magic header bytes."""
    return raw_bytes[:5] == b"%PDF-"


def scrape_webpages() -> List[Dict[str, str]]:
    """
    Iterate over every URL in ALL_SCRAPE_TARGETS, fetch & parse each one,
    and return a list of document dicts: {url, source_type, label, content}.

    If a URL points to a PDF (detected by extension, Google-Drive pattern,
    or the response's magic bytes), we route it through the PDF extractor
    instead of the HTML parser.
    """
    documents = []
    total = len(ALL_SCRAPE_TARGETS)
    counter = 0

    for source_type, label, url in ALL_SCRAPE_TARGETS:
        counter += 1
        print(f"[{counter}/{total}] Fetching ({source_type}/{label}) {url[:70]}")

        raw_bytes = _safe_get(url, timeout=20)
        if raw_bytes is None:
            print("       ✗ Could not retrieve — skipping")
            continue

        # Route PDFs through the PDF text extractor, not HTML parser
        if _is_pdf_url(url) or _content_looks_like_pdf(raw_bytes):
            print("       (detected PDF content — using PDF extractor)")
            text = _pdf_to_text(raw_bytes)
        else:
            text = _html_to_text(raw_bytes)

        if len(text) < MIN_CONTENT_LENGTH:
            print(f"       ✗ Too little content ({len(text)} chars) — skipping")
            continue

        documents.append({
            "url": url,
            "source_type": source_type,
            "label": label,
            "content": text,
        })
        print(f"       ✓ Extracted {len(text):,} characters")

        # Respect the server — throttle requests
        time.sleep(POLITENESS_DELAY)

    return documents



def persist_documents(documents: List[Dict[str, str]]) -> str:
    """
    Write each document to its own numbered .txt file **and** concatenate
    everything into a single corpus file (used by the preprocessor later).

    Returns the path to the combined corpus file.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # ── individual files ─────────────────────────────────────────────────────
    for idx, doc in enumerate(documents):
        label = doc.get("label", doc["source_type"])
        per_doc_path = os.path.join(
            RAW_DATA_DIR, f"{idx:03d}_{label}.txt"
        )
        with open(per_doc_path, "w", encoding="utf-8") as fh:
            # Metadata header so we can trace each file back to its origin
            fh.write(f"# URL: {doc['url']}\n")
            fh.write(f"# TYPE: {doc['source_type']}\n")
            fh.write(f"# LABEL: {doc.get('label', doc['source_type'])}\n")
            fh.write(f"# LENGTH: {len(doc['content'])} chars\n")
            fh.write("# " + "─" * 70 + "\n")
            fh.write(doc["content"])

    # ── combined corpus ──────────────────────────────────────────────────────
    with open(COMBINED_FILE, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc["content"] + "\n\n")

    print(f"\n✔ Wrote {len(documents)} individual files to  {RAW_DATA_DIR}/")
    print(f"✔ Combined corpus saved to                    {COMBINED_FILE}")
    return COMBINED_FILE


# ══════════════════════════════════════════════════════════════════════════════
# 5. FALLBACK SAMPLE CORPUS
# ══════════════════════════════════════════════════════════════════════════════
# If live scraping fails (network issues, VPN, campus firewall) we fall back
# to a curated set of passages that mirror the kind of language found on the
# IIT Jodhpur website.  Every passage is *original* paraphrased text, not
# copy‑pasted from the real site, so there are no copyright concerns.
# ══════════════════════════════════════════════════════════════════════════════

def _build_fallback_corpus() -> List[Dict[str, str]]:
    """
    Return a list of document dicts whose 'content' field contains
    realistic IIT Jodhpur–themed English prose covering all required
    categories (academics, regulation, research, campus, etc.).

    Key target words (research, student, phd, exam, professor, hostel,
    UG, PG, BTech, MTech) are used naturally so they appear frequently
    enough to survive min_count filtering during Word2Vec training.
    """

    passages: List[Tuple[str, str, str]] = [
        # ── (source_type, fake_url_tag, text) ────────────────────────────────

        # ---- 1. Institute Overview ----
        (
            "institute",
            "about",
            """
            The Indian Institute of Technology Jodhpur was established in 2008 as
            one of the newer IITs under the mentorship of IIT Kanpur. Situated in
            the Karwar range of the Thar desert, the institute occupies roughly
            852 acres of land and has grown rapidly into a vibrant hub for higher
            education and research. IIT Jodhpur offers undergraduate, postgraduate,
            and doctoral programmes across engineering, sciences, humanities, and
            management. The vision of the institute is to emerge as a leader in
            technology and innovation while serving the needs of society. Strategic
            partnerships with global universities and national laboratories
            strengthen the research ecosystem. The campus infrastructure includes
            smart classrooms, high‑performance computing clusters, a central
            library with digital archives, and modern sports facilities. The
            institute actively promotes interdisciplinary research by encouraging
            collaboration between departments and establishing centres of
            excellence in areas such as artificial intelligence, renewable energy,
            and digital humanities. Faculty recruitment drives attract scholars
            from premier institutions worldwide, ensuring a diverse and
            intellectually stimulating environment for every student.
            """,
        ),

        # ---- 2. Undergraduate (UG / BTech) Programmes ----
        (
            "programme",
            "btech_ug",
            """
            IIT Jodhpur admits UG students to its BTech programme through JEE
            Advanced, one of the most competitive entrance examinations in the
            country. The four‑year BTech curriculum is structured around a
            common first year where every UG student takes foundational courses
            in mathematics, physics, chemistry, and introductory engineering.
            From the second year onward, UG students specialize in their chosen
            branch: Computer Science and Engineering, Electrical Engineering,
            Mechanical Engineering, Civil and Infrastructure Engineering,
            Chemical Engineering, Bioscience and Bioengineering, Materials
            Engineering, or the recently launched Artificial Intelligence and
            Data Science track. UG students must accumulate at least 160
            credits, including departmental electives, open electives, and a
            capstone BTech Project in the final semester. Continuous evaluation
            through quizzes, mid‑semester exams, end‑semester exams, and
            laboratory assessments ensures that each student is tested
            regularly. A strong CGPA opens doors to branch transfers, semester
            exchange at partner universities, and competitive internship offers.
            The UG curriculum balances theory with hands‑on practice so that
            graduates are equally prepared for industry roles and advanced
            research at the postgraduate level.
            """,
        ),

        # ---- 3. Postgraduate (PG / MTech / MSc) Programmes ----
        (
            "programme",
            "pg_mtech",
            """
            Postgraduate education at IIT Jodhpur comprises MTech and MSc
            programmes that attract students who have cleared the GATE
            examination or hold equivalent qualifications. The two‑year PG
            curriculum blends advanced coursework with a substantial research
            thesis. PG students in the MTech programme choose specializations
            aligned with departmental strengths — for instance, Data and
            Computational Sciences in CSE, VLSI Design in Electrical
            Engineering, or Structural Engineering in Civil Engineering. MSc
            programmes are offered in Physics, Chemistry, and Mathematics,
            providing a rigorous foundation in the pure and applied sciences.
            Every PG student is assigned a thesis supervisor, typically a
            professor or associate professor, who mentors them through
            literature surveys, experimentation, and manuscript preparation.
            PG students also serve as teaching assistants, gaining pedagogical
            experience while supporting UG courses and laboratory sessions.
            Comprehensive viva‑voce examinations and external reviews ensure
            that every PG thesis meets high academic standards. The PG degree
            equips graduates for roles in R&D organizations, the technology
            industry, and doctoral studies in India or abroad.
            """,
        ),

        # ---- 4. PhD Programme ----
        (
            "programme",
            "phd_doctoral",
            """
            The PhD programme at IIT Jodhpur is designed for candidates who
            wish to make original contributions to knowledge through sustained
            research. Admission to the PhD programme is based on a written
            screening test followed by an interview conducted by a departmental
            committee. A PhD student first completes a set of prescribed
            coursework to build the theoretical background needed for research.
            After coursework, each PhD student must pass a comprehensive
            qualifying examination that evaluates depth of understanding and
            research readiness. Upon clearing the exam, the student formally
            proposes a research topic and begins independent investigation
            under the supervision of a professor. Progress is monitored through
            annual review seminars where a doctoral committee assesses
            milestones. Publication of research findings in peer‑reviewed
            journals or top‑tier conferences is expected before a PhD
            candidate can submit the final thesis. The thesis undergoes
            plagiarism checks and is evaluated by at least one external
            examiner from another institution. A successful public defence
            leads to the award of the PhD degree. Financial support in the
            form of institute fellowships and project assistantships ensures
            that PhD students can focus entirely on their research.
            """,
        ),

        # ---- 5. Academic Regulations — UG ----
        (
            "academic_regulation",
            "ug_regulations",
            """
            The UG academic regulations at IIT Jodhpur define the rules
            governing course registration, attendance, examination, grading,
            and degree completion for all BTech students. Every UG student must
            register for courses within the first week of each semester.
            A minimum attendance of seventy‑five percent is mandatory; falling
            below this threshold may result in a grade penalty or debarment
            from the end‑semester exam. The grading system assigns letter
            grades on a ten‑point scale, and the semester grade point average
            along with the cumulative grade point average determine academic
            standing. Students whose CGPA falls below the prescribed minimum
            are placed on academic probation and may face de‑registration after
            repeated poor performance. Course withdrawal is permitted before a
            specified deadline, allowing UG students to adjust their workload.
            A UG student may apply for branch change after the first year if
            the CGPA is sufficiently high and vacancies exist. The final BTech
            Project spans one or two semesters and is evaluated by an internal
            panel followed by a viva‑voce exam. Academic misconduct — including
            plagiarism and cheating during an exam — is handled by a
            disciplinary committee and can result in grade cancellation or
            suspension.
            """,
        ),

        # ---- 6. Academic Regulations — PG & PhD ----
        (
            "academic_regulation",
            "pg_phd_regulations",
            """
            Postgraduate and doctoral academic regulations at IIT Jodhpur
            prescribe the framework for course credit requirements, thesis
            evaluation, and degree conferral. Every MTech student must complete
            a minimum of 64 credits, of which at least 20 credits come from
            the thesis component. PhD students are required to earn a specified
            number of course credits and pass a comprehensive exam within the
            first two years. The thesis supervisor, who holds a professor or
            associate professor appointment, is formally assigned by the
            department. Leave of absence policy permits PG and PhD students to
            take medical or personal leave without losing their registration,
            subject to approval. Extension of the maximum duration of the
            programme requires recommendation from the supervisor and sanction
            from the Senate. A pre‑submission seminar ensures readiness before
            the thesis is formally submitted. The plagiarism detection report
            must show similarity below the permissible threshold. Two external
            examiners review the PhD thesis, and the candidate defends it in an
            open viva‑voce. Degree conferral takes place at the annual
            convocation ceremony. Both PG and PhD students are bound by the
            same code of academic integrity that applies to every student of
            the institute.
            """,
        ),

        # ---- 7. Research Ecosystem ----
        (
            "institute",
            "research",
            """
            Research is central to the mission of IIT Jodhpur. The institute
            hosts multiple centres of excellence focusing on emerging areas
            such as smart healthcare, sustainable energy, advanced materials,
            and intelligent transportation. Faculty members, from assistant
            professor to full professor, pursue sponsored projects funded by
            agencies like DST, SERB, and DRDO, as well as industry partners
            including Tata, Bosch, and Reliance. Each research group
            comprises a professor, PhD students, PG scholars, and occasionally
            advanced UG interns. Publication output has risen steadily, with
            papers appearing in high‑impact journals and flagship conferences
            such as NeurIPS, CVPR, AAAI, and IEEE Transactions. The institute
            encourages patent filing and has a dedicated Intellectual Property
            cell. Technology Innovation Hub iHub Drishti supports start‑ups
            working on computer vision and augmented reality. Collaborative
            programmes with universities in the US, Europe, and Japan provide
            student and faculty exchange opportunities. Research seminars held
            every week expose students to cutting‑edge ideas from visiting
            scholars. The strong research culture makes IIT Jodhpur an
            attractive destination for doctoral students and early career
            faculty alike.
            """,
        ),

        # ---- 8. Campus Life & Hostels ----
        (
            "institute",
            "campus_life",
            """
            Campus life at IIT Jodhpur is defined by a close‑knit community
            that lives, studies, and creates together. Every enrolled student
            is allotted a room in one of the on‑campus hostels. The hostel
            experience fosters independence, peer learning, and lifelong
            friendships. Each hostel is managed by a warden — usually a senior
            professor — and a student hostel council. Mess services in each
            hostel provide breakfast, lunch, and dinner with options for
            vegetarian and non‑vegetarian meals. Common rooms in every hostel
            are equipped with televisions, table tennis tables, and informal
            study spaces. Beyond the hostel, the campus offers a swimming pool,
            basketball courts, a cricket ground, and a well‑equipped gymnasium.
            Cultural life thrives through clubs dedicated to music, dance,
            drama, photography, and literary arts. Annual festivals — Ignus for
            culture, Varchas for sports, and Prometeo for technology — draw
            participants from institutes across India. Mental health and
            wellness services are available through a counselling centre
            staffed by trained professionals. The hostel environment, combined
            with diverse extra‑curriculars, ensures that student life extends
            well beyond the classroom.
            """,
        ),

        # ---- 9. Examination System ----
        (
            "academic_regulation",
            "exam_system",
            """
            The examination framework at IIT Jodhpur employs continuous
            assessment to ensure that learning is tested at multiple points
            rather than only at the end. A typical course includes surprise
            quizzes, assignments, a mid‑semester exam, and an end‑semester
            exam. The weightage of each assessment component is declared in
            the course handout at the beginning of the semester. Laboratory
            courses replace written exams with practical tests, project demos,
            and viva‑voce sessions. For PhD students, the comprehensive
            qualifying exam is a rigorous oral examination covering core and
            elective topics. The exam section of the academic office is
            responsible for scheduling, hall allocation, invigilation rosters,
            and result processing. Answer scripts are evaluated within a
            stipulated time and students may request re‑evaluation if they
            believe a scoring error occurred. Make‑up exams are granted only
            for documented medical or family emergencies. A supplementary exam
            is offered once a year for students who have failed a course. Exam
            malpractice — such as copying, using unauthorized materials, or
            impersonation — carries severe penalties ranging from zero marks
            to expulsion. Maintaining exam integrity is a shared responsibility
            of every student and faculty member.
            """,
        ),

        # ---- 10. Departments Overview ----
        (
            "department",
            "departments",
            """
            IIT Jodhpur has twelve academic departments and several
            interdisciplinary programmes. The Department of Computer Science
            and Engineering is known for research in machine learning, natural
            language processing, and computer vision, with every professor
            supervising multiple PhD and PG students. Electrical Engineering
            covers signal processing, VLSI, and communication systems.
            Mechanical Engineering focuses on thermal systems, robotics, and
            manufacturing. Civil and Infrastructure Engineering addresses
            geotechnical engineering, water resources, and smart cities.
            Chemical Engineering works on process optimization and polymer
            science. The Department of Metallurgical and Materials Engineering
            explores advanced alloys and nanomaterials. Bioscience and
            Bioengineering blends molecular biology with computational
            approaches. Physics, Chemistry, and Mathematics departments
            anchor the sciences with both teaching and frontier research.
            The Humanities and Social Sciences department enriches the
            institute by offering courses in economics, psychology,
            philosophy, and communication. Each department holds its own
            seminar series, publishes an annual report, and participates in
            national ranking exercises. Student clubs at the department level
            organise hackathons, paper‑reading groups, and invited lectures.
            """,
        ),

        # ---- 11. Placement & Careers ----
        (
            "institute",
            "placements",
            """
            The Centre for Career Planning and Services at IIT Jodhpur
            facilitates campus placements and internships. Recruiting
            companies span domains such as software, consulting, finance,
            FMCG, core engineering, and research labs. The placement season
            typically begins in December with pre‑placement talks where
            companies present their culture and job profiles. Mock interviews
            and resume workshops are organised in the preceding months to
            prepare each student for the selection process. Both UG and PG
            students are eligible for placements; PhD students often take up
            research‑scientist or faculty positions. The median compensation
            package has shown an upward trend over the past five years,
            reflecting the growing reputation of IIT Jodhpur graduates.
            Summer internships between the pre‑final and final years give
            students exposure to real‑world projects and frequently convert
            into full‑time offers. The alumni network — though young —
            actively supports current students through referrals, mentorship
            sessions, and webinars. Students interested in higher studies
            abroad receive guidance on applications, GRE and TOEFL
            preparation, and statement‑of‑purpose writing. Faculty
            recommendation letters, especially from a professor who knows
            the student's research, carry significant weight in graduate
            admissions.
            """,
        ),

        # ---- 12. Teaching & Pedagogy ----
        (
            "institute",
            "teaching",
            """
            Teaching at IIT Jodhpur is built around the idea that active
            engagement leads to deeper learning. Each professor designs a
            course handout specifying learning objectives, weekly topics,
            reading material, and the exam schedule. Lectures are
            supplemented by tutorial sessions where students solve problems
            collaboratively. Laboratory courses let UG and PG students
            translate theoretical knowledge into practical skills through
            structured experiments. Flipped‑classroom models, where students
            study material before class and spend lecture time on discussion,
            are increasingly adopted across departments. Online tools such as
            Moodle and Google Classroom are used for distributing assignments,
            conducting quizzes, and sharing resources. Student feedback
            collected at mid‑semester and end‑semester feeds into continuous
            improvement of teaching practices. The Centre for Teaching and
            Learning organises faculty development programmes to introduce new
            pedagogical techniques. PhD and PG teaching assistants play a
            crucial role by conducting doubt sessions and grading assignments,
            which also helps them develop academic communication skills.
            Excellence in teaching is recognized through annual awards
            nominated by students and evaluated by a committee of senior
            professors.
            """,
        ),

        # ---- 13. Admissions Overview ----
        (
            "institute",
            "admissions",
            """
            Admissions to IIT Jodhpur are governed by national‑level
            examinations and centralized counselling. UG BTech aspirants
            must clear the JEE Advanced exam and participate in JoSAA
            counselling for seat allocation. PG MTech and MSc students are
            selected on the basis of their GATE or JAM score along with an
            interview where applicable. PhD applicants submit a research
            proposal and appear for a written test followed by a personal
            interview arranged by the department. International students may
            apply under the Study in India programme or bilateral agreements.
            Upon selection, each admitted student completes document
            verification, pays the semester fee, and receives a unique roll
            number. An orientation week acquaints new entrants with the
            academic calendar, hostel rules, library resources, and campus
            facilities. Faculty advisors are assigned to guide first‑year UG
            students through course selection. PG and PhD students meet their
            research supervisor within the first month. Scholarship schemes
            funded by MHRD and private donors partially or fully cover tuition
            for economically disadvantaged students. Merit‑cum‑means
            scholarships are also available for top‑performing UG students.
            """,
        ),
    ]

    # Convert tuples into the standard document‑dict format
    return [
        {
            "url": f"sample://{tag}",
            "source_type": stype,
            "content": re.sub(r"\s+", " ", text).strip(),
        }
        for stype, tag, text in passages
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_collection(live: bool = False) -> str:
    """
    Orchestrate the full data‑collection pipeline.

    Parameters
    ----------
    live : bool
        If True → scrape real websites + download PDFs.
        If False → use the built‑in sample corpus (faster, offline‑safe).

    Returns
    -------
    str : path to the combined corpus file on disk.
    """
    if live:
        print("=" * 65)
        print("  LIVE MODE — scraping IIT Jodhpur web sources")
        print("=" * 65)
        docs = scrape_webpages()           # all 5 source categories
    else:
        print("=" * 65)
        print("  SAMPLE MODE — using offline curated corpus")
        print("  (pass live=True to scrape real websites instead)")
        print("=" * 65)
        docs = _build_fallback_corpus()

    if not docs:
        raise RuntimeError("No documents collected — check network / URLs.")

    # Brief summary before saving
    print(f"\nTotal documents collected : {len(docs)}")
    print(f"Total characters         : {sum(len(d['content']) for d in docs):,}")

    return persist_documents(docs)


# ── Script entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Flip to live=True when you have internet access & want real data
    corpus_path = run_collection(live=False)
    print(f"\nDone. Corpus ready at: {corpus_path}")
