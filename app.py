import tempfile
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from resume_analyzer import analyze_resume, read_resume_from_file
from resume_badges import badge_for_score
from resume_improver import improve_resume


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_uploaded_resume(uploaded_file):
    """Extract and analyze one uploaded resume, returning a result or error."""
    if uploaded_file is None or uploaded_file.filename == "":
        return None, "Choose a resume file to analyze."
    if not allowed_file(uploaded_file.filename):
        return None, "Upload a TXT, PDF, PNG, JPG, WEBP, BMP, or TIFF resume."

    filename = secure_filename(uploaded_file.filename)
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        uploaded_file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        resume_text = read_resume_from_file(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if not resume_text.strip():
        return None, "I could not extract text from that file. For image resumes, install OCR support and make sure Tesseract is available."

    result = analyze_resume(resume_text)
    result["filename"] = filename
    result["badge"] = badge_for_score(result["score"])
    if result["score"] < 6:
        result["improved_resume"] = improve_resume(resume_text)
    return result, None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        result, error = analyze_uploaded_resume(request.files.get("resume"))

    return render_template("index.html", result=result, error=error)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    results = None
    error = None

    if request.method == "POST":
        first_result, first_error = analyze_uploaded_resume(request.files.get("resume_one"))
        second_result, second_error = analyze_uploaded_resume(request.files.get("resume_two"))
        if first_error or second_error:
            error = first_error or second_error
        else:
            results = [first_result, second_result]

    return render_template("compare.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)
