import tempfile
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from resume_analyzer import analyze_resume, read_resume_from_file


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        if uploaded_file is None or uploaded_file.filename == "":
            error = "Choose a resume file to analyze."
        elif not allowed_file(uploaded_file.filename):
            error = "Upload a TXT, PDF, PNG, JPG, WEBP, BMP, or TIFF resume."
        else:
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
                error = "I could not extract text from that file. For image resumes, install OCR support and make sure Tesseract is available."
            else:
                result = analyze_resume(resume_text)
                result["filename"] = filename

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
