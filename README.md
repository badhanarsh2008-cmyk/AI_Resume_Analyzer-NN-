# Resume Analyzer

A simple Python resume analyzer that scores a resume from 1 to 10 and gives quick feedback on strong areas and improvements.

## Features

- Reads resume text from a `.txt` file
- Extracts text from a `.pdf` file
- Accepts image resumes with optional OCR support
- Accepts pasted resume text from the terminal
<<<<<<< HEAD
- Provides a web upload page with score, key points, missing areas, and a 20-point feature graph with hover explanations
=======
- Provides a web upload page with score, key points, missing areas, and a feature graph
>>>>>>> 55e9bcd21875a652499e5513d581cca63b564100
- Generates an ATS-friendly improved-resume draft below the analysis when the score is under 6
- Checks role-relevant skills, sections, contact details, action verbs, measurable results, and readability
- Uses a small neural network to generate a resume score
- Saves trained model weights in `resume_model.json` so later analyses are faster

## Requirements

Python 3 is required.

Install the web app dependencies:

```bash
pip install -r requirements.txt
```

For image OCR, `pytesseract` also needs the Tesseract OCR program installed on your computer.

The first analysis trains the model and creates `resume_model.json`. Later
analyses load this file instead of training again. When deploying, keep this
file in persistent storage if available. If it is missing, the app trains and
saves a replacement model automatically.

## How to Run

Start the web app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Analyze a text resume:

```bash
python resume_analyzer.py sample_resume.txt
```

Analyze a PDF resume:

```bash
python resume_analyzer.py sample_resume.pdf
```

Run without a file and paste resume text:

```bash
python resume_analyzer.py
```

## Output

The program prints:

- Resume score out of 10
- Strong areas
- Areas to improve

The web-only resume-draft feature lives in `resume_improver.py`, so its formatting and improvement rules can be expanded independently of `app.py` and the scoring model.

Example:

```text
Resume score: 7/10
Strong areas: Technical skills, Education, Important sections, Contact details
Improve: Resume length, Vocabulary range
```
