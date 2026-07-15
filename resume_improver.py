"""Create a conservative, ATS-friendly resume draft from extracted resume text.

This module deliberately does not invent employers, dates, skills, or metrics.  It
only reorganizes the candidate's supplied text and clearly marks information the
candidate needs to add.
"""

import re


SECTION_HEADINGS = {
    "summary": "PROFESSIONAL SUMMARY",
    "profile": "PROFESSIONAL SUMMARY",
    "objective": "PROFESSIONAL SUMMARY",
    "experience": "EXPERIENCE",
    "work experience": "EXPERIENCE",
    "work history": "EXPERIENCE",
    "employment": "EXPERIENCE",
    "projects": "PROJECTS",
    "education": "EDUCATION",
    "skills": "SKILLS",
    "technical skills": "SKILLS",
    "certifications": "CERTIFICATIONS",
    "certificates": "CERTIFICATIONS",
    "achievements": "ACHIEVEMENTS",
}


def _clean_lines(resume_text):
    """Return useful lines while removing repeated whitespace and page labels."""
    lines = []
    for line in resume_text.splitlines():
        line = re.sub(r"\s+", " ", line).strip(" -•\t")
        if line and not re.fullmatch(r"page \d+( of \d+)?", line, flags=re.I):
            lines.append(line)
    return lines


def _is_contact_line(line):
    return bool(re.search(r"@[\w.-]+|linkedin|github|\+?\d[\d() .-]{7,}\d", line, re.I))


def _is_heading(line):
    normalized = line.lower().rstrip(":")
    return normalized in SECTION_HEADINGS


def _bullets(lines):
    if not lines:
        return ["[Add relevant details here.]"]
    return [f"• {line}" for line in lines]


def improve_resume(resume_text):
    """Return a structured resume draft based only on the uploaded content."""
    lines = _clean_lines(resume_text)
    if not lines:
        return "Unable to create a draft because no resume text was extracted."

    name = lines[0]
    remaining = lines[1:]
    contacts = [line for line in remaining if _is_contact_line(line)]
    content = [line for line in remaining if line not in contacts]

    sections = {}
    current = "UNSORTED EXPERIENCE OR PROJECT DETAILS"
    sections[current] = []
    for line in content:
        if _is_heading(line):
            current = SECTION_HEADINGS[line.lower().rstrip(":")]
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    summary = sections.pop("PROFESSIONAL SUMMARY", [])
    if not summary:
        summary = [
            "[Write 2–3 lines naming your target role, strongest relevant skills, and the value you bring.]"
        ]

    output = [name.upper()]
    output.append(" | ".join(contacts) if contacts else "[Add phone] | [Add email] | [Add LinkedIn or portfolio]")
    output.extend(["", "PROFESSIONAL SUMMARY", *summary])

    preferred_order = ["SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION", "CERTIFICATIONS", "ACHIEVEMENTS"]
    for heading in preferred_order:
        if heading in sections:
            output.extend(["", heading, *_bullets(sections.pop(heading))])

    for heading, section_lines in sections.items():
        if section_lines:
            output.extend(["", heading, *_bullets(section_lines)])

    output.extend([
        "",
        "EDIT BEFORE SENDING",
        "• Replace bracketed prompts with your real information.",
        "• For each experience bullet, start with an action verb and add a result, number, or percentage where truthful.",
        "• Keep only skills and claims you can support in an interview.",
    ])
    return "\n".join(output)
