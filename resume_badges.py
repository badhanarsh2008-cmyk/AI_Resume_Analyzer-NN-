"""Friendly score labels used by the resume-analysis result page."""


def badge_for_score(score):
    """Return one modern, encouraging badge for a 1-to-10 resume score."""
    if score >= 8:
        return {
            "tone": "excellent",
            "icon": "✦",
            "label": "Excellent Profile",
            "message": "Polished, complete, and ready to stand out.",
        }
    if score >= 6:
        return {
            "tone": "good",
            "icon": "↗",
            "label": "Strong Foundation",
            "message": "A solid resume with room to make it even sharper.",
        }
    return {
        "tone": "refine",
        "icon": "✧",
        "label": "Ready to Refine",
        "message": "A few focused updates can make a meaningful difference.",
    }
