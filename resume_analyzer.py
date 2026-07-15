import json
import math
import random
import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


FEATURE_DETAILS = [
    ("Resume length", "Checks whether the resume has enough useful text without being excessively short."),
    ("Role-relevant skills", "Looks for technical and professional skills used across common roles."),
    ("Action verbs", "Counts strong verbs such as built, developed, led, and improved."),
    ("Education / training", "Looks for degrees, courses, certifications, or other training."),
    ("Important sections", "Checks for common sections such as Skills, Experience, Education, and Projects."),
    ("Contact details", "Checks for an email, phone number, and professional profile link."),
    ("Measurable results", "Looks for numbers or percentages that make achievements more specific."),
    ("Bullet structure", "Checks for bullet points that make experience easier to scan."),
    ("Readable lines", "Rewards resumes that avoid too many very long lines of text."),
    ("Vocabulary range", "Measures the variety of words used across the resume."),
    ("Target role clarity", "Looks for a clear job title or target role, such as engineer, analyst, or manager."),
    ("Professional summary", "Checks for a summary, profile, or objective near the start of the resume."),
    ("Project evidence", "Looks for a Projects section or project-related work to show practical experience."),
    ("Experience evidence", "Looks for work, internship, volunteer, or other experience information."),
    ("Portfolio links", "Checks for GitHub, LinkedIn, a portfolio, Kaggle, or another professional link."),
    ("Certifications", "Looks for a Certifications section or certification-related content."),
    ("Date clarity", "Looks for dates that help recruiters understand education and experience timelines."),
    ("Skills organization", "Checks whether skills are grouped under a clear skills or technologies section."),
    ("Achievement bullets", "Checks for bullets that combine an action with a specific outcome or result."),
    ("Completion check", "Rewards resumes without visible placeholders such as [Add your email] or TODO."),
]
FEATURE_LABELS = [label for label, _ in FEATURE_DETAILS]

ROLE_SKILL_KEYWORDS = [
    "accounting", "administration", "admissions", "air conditioning", "analytics", "assembly",
    "assessment", "audit", "auto repair", "automotive", "aws", "azure", "billing", "bookkeeping",
    "budgeting", "carpentry", "cash handling", "classroom", "client service", "coaching",
    "communication", "compliance", "computer", "construction", "content", "cooking",
    "counseling", "crm", "css", "curriculum", "customer service", "data entry", "database",
    "diagnostics", "django", "docker", "documentation", "electrical", "equipment",
    "excel", "fabrication", "first aid", "flask", "food safety", "forklift", "git",
    "grading", "guest service", "html", "hvac", "inventory", "java", "javascript",
    "kubernetes", "lesson planning", "linux", "logistics", "machine learning", "maintenance",
    "marketing", "mathematics", "mechanical", "merchandising", "microsoft office", "node",
    "nursing", "operations", "pandas", "patient care", "payroll", "plumbing", "pos",
    "preventive maintenance", "project management", "pytorch", "python", "quality control",
    "react", "records management", "recruiting", "repairs", "reporting", "research",
    "safety", "sales", "scheduling", "scikit", "sql", "stocking", "supervision",
    "teaching", "team leadership", "tensorflow", "training", "troubleshooting",
    "typescript", "welding", "writing"
]

MODEL_PATH = Path(__file__).with_name("resume_model.json")
MODEL_FORMAT_VERSION = 3
_trained_model = None


class NeuralNetwork:
    def __init__(self, input_size, hidden_size=12, learning_rate=0.08, seed=7):
        generator = random.Random(seed)
        self.learning_rate = learning_rate
        self.w1 = [[generator.uniform(-0.7, 0.7) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [generator.uniform(-0.7, 0.7) for _ in range(hidden_size)]
        self.b2 = 0.0

    def sigmoid(self, x):
        x = max(-60, min(60, x))
        return 1 / (1 + math.exp(-x))

    def forward(self, features):
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            value = sum(weight * feature for weight, feature in zip(row, features)) + bias
            hidden.append(self.sigmoid(value))
        output_raw = sum(weight * value for weight, value in zip(self.w2, hidden)) + self.b2
        output = self.sigmoid(output_raw)
        return hidden, output

    def train(self, data, epochs=700):
        training_data = list(data)
        shuffler = random.Random(21)
        for _ in range(epochs):
            shuffler.shuffle(training_data)
            for features, target in training_data:
                hidden, output = self.forward(features)
                error = output - target
                output_delta = error * output * (1 - output)
                old_w2 = self.w2[:]
                for i in range(len(self.w2)):
                    self.w2[i] -= self.learning_rate * output_delta * hidden[i]
                self.b2 -= self.learning_rate * output_delta
                for i in range(len(self.w1)):
                    hidden_delta = output_delta * old_w2[i] * hidden[i] * (1 - hidden[i])
                    for j in range(len(self.w1[i])):
                        self.w1[i][j] -= self.learning_rate * hidden_delta * features[j]
                    self.b1[i] -= self.learning_rate * hidden_delta

    def predict(self, features):
        return self.forward(features)[1]

    def to_dict(self):
        """Return the trained parameters in a JSON-safe format."""
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "input_size": len(self.w1[0]),
            "hidden_size": len(self.w1),
            "learning_rate": self.learning_rate,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, data, expected_input_size):
        """Create a network from saved parameters after basic validation."""
        if data.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("Unsupported model format")
        if data.get("input_size") != expected_input_size:
            raise ValueError("Saved model has a different input size")

        hidden_size = data.get("hidden_size")
        w1, b1, w2 = data.get("w1"), data.get("b1"), data.get("w2")
        if (
            not isinstance(hidden_size, int)
            or hidden_size < 1
            or not isinstance(w1, list)
            or len(w1) != hidden_size
            or any(not isinstance(row, list) or len(row) != expected_input_size for row in w1)
            or not isinstance(b1, list)
            or len(b1) != hidden_size
            or not isinstance(w2, list)
            or len(w2) != hidden_size
            or not isinstance(data.get("b2"), (int, float))
            or not isinstance(data.get("learning_rate", 0.08), (int, float))
            or any(not isinstance(value, (int, float)) for row in w1 for value in row)
            or any(not isinstance(value, (int, float)) for value in b1 + w2)
        ):
            raise ValueError("Saved model parameters are invalid")

        model = cls(expected_input_size, hidden_size, data.get("learning_rate", 0.08))
        model.w1 = w1
        model.b1 = b1
        model.w2 = w2
        model.b2 = data.get("b2", 0.0)
        return model


def count_keywords(text, keywords):
    return sum(
        1
        for keyword in keywords
        if re.search(r"(?<![a-z0-9+#.])" + re.escape(keyword) + r"(?![a-z0-9+#.])", text)
    )


def normalized(value, limit):
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, value / limit))


def extract_features(resume):
    text = resume.lower()
    words = re.findall(r"[a-zA-Z0-9+#.]+", text)
    lines = [line.strip() for line in resume.splitlines() if line.strip()]
    action_keywords = [
        "built", "created", "developed", "designed", "implemented", "improved", "optimized",
        "automated", "managed", "led", "launched", "deployed", "analyzed", "trained", "reduced",
        "taught", "mentored", "supervised", "coordinated", "maintained", "repaired", "installed",
        "diagnosed", "served", "supported", "organized", "prepared", "planned", "resolved",
        "delivered", "processed", "inspected", "operated", "assisted", "increased"
    ]
    education_keywords = [
        "bachelor", "master", "phd", "degree", "university", "college", "certification",
        "certified", "course", "diploma", "license", "licensed", "apprenticeship", "training",
        "workshop", "school", "board", "iti"
    ]
    section_keywords = [
        "experience", "projects", "skills", "education", "summary", "objective", "certifications",
        "achievements", "profile", "work history", "employment", "licenses", "training",
        "awards", "volunteer", "references"
    ]
    contact_score = int(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))) + int(bool(re.search(r"\b\d{10}\b|\(\d{3}\)\s*\d{3}[- ]?\d{4}", text))) + int("linkedin" in text or "github" in text)
    number_score = len(re.findall(r"\b\d+%?\b", text))
    bullet_score = resume.count("\n-") + resume.count("\n*") + resume.count(chr(8226))
    long_line_penalty = sum(1 for line in lines if len(line) > 140)
    target_role_keywords = [
        "engineer", "developer", "analyst", "designer", "manager", "specialist", "consultant",
        "coordinator", "scientist", "intern", "technician", "teacher", "accountant", "nurse"
    ]
    date_matches = re.findall(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{4}\b|\b(?:19|20)\d{2}\b",
        text,
    )
    portfolio_score = int("linkedin" in text) + int("github" in text) + int(bool(re.search(r"\b(?:portfolio|kaggle|gitlab|behance)\b", text)))
    skills_organization = (
        int(bool(re.search(r"\b(?:technical )?skills\b|\btechnologies\b", text)))
        + int(bool(re.search(r"\b(?:technical )?skills\b[^\n]{0,60}[:|]", text)))
        + int(count_keywords(text, ROLE_SKILL_KEYWORDS) >= 5)
    )
    achievement_bullets = sum(
        1
        for line in lines
        if re.match(r"^(?:[-*•]\s*)?(?:" + "|".join(action_keywords) + r")\b", line.lower())
        and re.search(r"\b\d+%?\b|(?:increased|reduced|improved|saved|grew)", line.lower())
    )
    placeholder_count = len(re.findall(r"\[[^\]]*(?:add|insert|your)[^\]]*\]|\b(?:todo|tbd)\b", text))
    features = [
        normalized(len(words), 650),
        normalized(count_keywords(text, ROLE_SKILL_KEYWORDS), 12),
        normalized(count_keywords(text, action_keywords), 10),
        normalized(count_keywords(text, education_keywords), 5),
        normalized(count_keywords(text, section_keywords), 7),
        normalized(contact_score, 3),
        normalized(number_score, 8),
        normalized(bullet_score, 12),
        1 - normalized(long_line_penalty, 6),
        normalized(len(set(words)), 350),
        normalized(count_keywords(text, target_role_keywords), 2),
        float(bool(re.search(r"\b(?:summary|profile|objective)\b", text))),
        float(bool(re.search(r"\bprojects?\b|\bportfolio\b", text))),
        float(bool(re.search(r"\b(?:experience|employment|work history|internship|volunteer)\b", text))),
        normalized(portfolio_score, 2),
        float(bool(re.search(r"\b(?:certifications?|certified|certificate)\b", text))),
        normalized(len(date_matches), 3),
        normalized(skills_organization, 3),
        normalized(achievement_bullets, 3),
        1 - normalized(placeholder_count, 1),
    ]
    return features


def rule_score(features):
    weights = [
        1.0, 1.7, 1.3, 0.8, 1.2, 1.0, 1.1, 0.8, 0.6, 0.7,
        0.9, 1.1, 1.0, 1.2, 0.7, 0.6, 0.7, 0.8, 1.2, 0.8,
    ]
    value = sum(feature * weight for feature, weight in zip(features, weights)) / sum(weights)
    return max(0.0, min(1.0, value))


def make_training_data():
    data = []
    generator = random.Random(21)
    for _ in range(90):
        features = [generator.random() for _ in FEATURE_LABELS]
        target = rule_score(features)
        data.append((features, target))
    examples = [
        ([0.05, 0.02, 0.01, 0.0, 0.05, 0.0, 0.0, 0.0, 0.7, 0.05] + [0.0] * 9 + [0.3], 0.05),
        ([0.25, 0.2, 0.15, 0.2, 0.3, 0.4, 0.1, 0.2, 0.8, 0.25] + [0.2] * 9 + [0.7], 0.32),
        ([0.55, 0.55, 0.5, 0.45, 0.65, 0.7, 0.45, 0.5, 0.9, 0.55] + [0.55] * 9 + [1.0], 0.62),
        ([0.85, 0.9, 0.85, 0.7, 0.85, 1.0, 0.9, 0.8, 0.95, 0.85] + [0.85] * 9 + [1.0], 0.92)
    ]
    data.extend(examples)
    return data


def load_or_train_model(input_size):
    """Load the cached model, or train and save it if it does not exist yet."""
    global _trained_model
    if _trained_model is not None and len(_trained_model.w1[0]) == input_size:
        return _trained_model

    try:
        saved_model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        _trained_model = NeuralNetwork.from_dict(saved_model, input_size)
        return _trained_model
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A missing, invalid, or older model is safely replaced by a new one.
        model = NeuralNetwork(input_size)
        model.train(make_training_data())
        temporary_model_path = MODEL_PATH.with_suffix(".tmp")
        temporary_model_path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
        temporary_model_path.replace(MODEL_PATH)
        _trained_model = model
        return _trained_model


def score_resume(resume):
    features = extract_features(resume)
    network = load_or_train_model(len(features))
    raw_score = network.predict(features)
    score = max(1, min(10, round(raw_score * 9 + 1)))
    return score, features


def feedback(features):
    strengths = [label for label, value in zip(FEATURE_LABELS, features) if value >= 0.65]
    improvements = [label for label, value in zip(FEATURE_LABELS, features) if value < 0.45]
    return strengths[:4], improvements[:4]


def analyze_resume(resume):
    score, features = score_resume(resume)
    strengths, improvements = feedback(features)
    feature_scores = [
        {"label": label, "description": description, "value": round(value * 100)}
        for (label, description), value in zip(FEATURE_DETAILS, features)
    ]
    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "features": feature_scores,
        "word_count": len(re.findall(r"[a-zA-Z0-9+#.]+", resume))
    }


def read_resume_from_file(path):
    file_path = Path(path)
    if not file_path.exists():
        print("File not found.")
        return ""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        return extract_text_from_image(file_path)
    print("Only .txt, .pdf, and image files are supported.")
    return ""


def extract_text_from_pdf(file_path):
    if PdfReader is None:
        print("PDF support requires pypdf. Install it with: pip install pypdf")
        return ""
    try:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        print(f"Could not read PDF: {error}")
        return ""
    return "\n".join(pages)


def extract_text_from_image(file_path):
    if Image is None or pytesseract is None:
        print("Image support requires Pillow and pytesseract. Install them with: pip install pillow pytesseract")
        return ""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as error:
        print(f"Could not read image: {error}")
        return ""


def read_resume_from_paste():
    print("Paste your resume text. Press Enter on an empty line to analyze.")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def read_resume():
    if len(sys.argv) > 1:
        return read_resume_from_file(sys.argv[1])
    source = input("Enter txt/pdf/image file path or press Enter to paste resume: ").strip().strip('"')
    if source:
        return read_resume_from_file(source)
    return read_resume_from_paste()


def main():
    resume = read_resume()
    if not resume.strip():
        print("No resume text entered.")
        return
    score, features = score_resume(resume)
    strengths, improvements = feedback(features)
    print(f"\nResume score: {score}/10")
    print("Strong areas: " + (", ".join(strengths) if strengths else "None detected"))
    print("Improve: " + (", ".join(improvements) if improvements else "Looks balanced"))


if __name__ == "__main__":
    main()
