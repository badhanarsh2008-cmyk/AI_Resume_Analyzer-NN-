# Resume Analyzer

A simple Python resume analyzer that scores a resume from 1 to 10 and gives quick feedback on strong areas and improvements.

## Features

- Reads resume text from a `.txt` file
- Extracts text from a `.pdf` file
- Accepts pasted resume text from the terminal
- Checks skills, sections, contact details, action verbs, measurable results, and readability
- Uses a small neural network to generate a resume score

## Requirements

Python 3 is required.

For PDF support, install `pypdf`:

```bash
pip install pypdf
```

## How to Run

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

Example:

```text
Resume score: 7/10
Strong areas: Technical skills, Education, Important sections, Contact details
Improve: Resume length, Vocabulary range
```
