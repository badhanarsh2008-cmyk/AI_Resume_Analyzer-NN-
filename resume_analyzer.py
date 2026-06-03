import math
import re
import random
import sys
from pathlib import Path


class NeuralNetwork:
    def __init__(self, input_size, hidden_size=12, learning_rate=0.08, seed=7):
        random.seed(seed)
        self.learning_rate = learning_rate
        self.w1 = [[random.uniform(-0.7, 0.7) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [random.uniform(-0.7, 0.7) for _ in range(hidden_size)]
        self.b2 = 0.0

    def sigmoid(self, x):
        x = max(-60, min(60, x))
        return 1 / (1 + math.exp(-x))

    def forward(self, features):
        hidden_raw = []
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            value = sum(weight * feature for weight, feature in zip(row, features)) + bias
            hidden_raw.append(value)
            hidden.append(self.sigmoid(value))
        output_raw = sum(weight * value for weight, value in zip(self.w2, hidden)) + self.b2
        output = self.sigmoid(output_raw)
        return hidden, output

    def train(self, data, epochs=700):
        for _ in range(epochs):
            random.shuffle(data)
            for features, target in data:
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


def count_keywords(text, keywords):
    return sum(1 for keyword in keywords if re.search(r"\b" + re.escape(keyword) + r"\b", text))


def normalized(value, limit):
    return max(0.0, min(1.0, value / limit))


def extract_features(resume):
    text = resume.lower()
    words = re.findall(r"[a-zA-Z0-9+#.]+", text)
    technical_keywords = [
        "python", "java", "javascript", "typescript", "c++", "sql", "react", "node", "django",
        "flask", "tensorflow", "pytorch", "scikit", "pandas", "numpy", "aws", "azure", "docker",
        "kubernetes", "git", "linux", "html", "css", "machine", "learning", "api", "excel"
    ]
    action_keywords = [
        "built", "created", "developed", "designed", "implemented", "improved", "optimized",
        "automated", "managed", "led", "launched", "deployed", "analyzed", "trained", "reduced"
    ]
    education_keywords = [
        "bachelor", "master", "phd", "degree", "university", "college", "certification",
        "certified", "course", "diploma"
    ]
    section_keywords = [
        "experience", "projects", "skills", "education", "summary", "objective", "certifications",
        "achievements"
    ]
    contact_score = int(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))) + int(bool(re.search(r"\b\d{10}\b|\(\d{3}\)\s*\d{3}[- ]?\d{4}", text))) + int("linkedin" in text or "github" in text)
    number_score = len(re.findall(r"\b\d+%?\b", text))
    bullet_score = resume.count("\n-") + resume.count("\n*") + resume.count(chr(8226))
    long_line_penalty = sum(1 for line in resume.splitlines() if len(line) > 140)
    features = [
        normalized(len(words), 650),
        normalized(count_keywords(text, technical_keywords), 16),
        normalized(count_keywords(text, action_keywords), 10),
        normalized(count_keywords(text, education_keywords), 5),
        normalized(count_keywords(text, section_keywords), 7),
        normalized(contact_score, 3),
        normalized(number_score, 8),
        normalized(bullet_score, 12),
        1 - normalized(long_line_penalty, 6),
        normalized(len(set(words)), 350)
    ]
    return features


def rule_score(features):
    weights = [1.0, 1.7, 1.3, 0.8, 1.2, 1.0, 1.1, 0.8, 0.6, 0.7]
    value = sum(feature * weight for feature, weight in zip(features, weights)) / sum(weights)
    return max(0.0, min(1.0, value))


def make_training_data():
    data = []
    random.seed(21)
    for _ in range(90):
        features = [random.random() for _ in range(10)]
        target = rule_score(features)
        data.append((features, target))
    examples = [
        ([0.05, 0.02, 0.01, 0.0, 0.05, 0.0, 0.0, 0.0, 0.7, 0.05], 0.05),
        ([0.25, 0.2, 0.15, 0.2, 0.3, 0.4, 0.1, 0.2, 0.8, 0.25], 0.32),
        ([0.55, 0.55, 0.5, 0.45, 0.65, 0.7, 0.45, 0.5, 0.9, 0.55], 0.62),
        ([0.85, 0.9, 0.85, 0.7, 0.85, 1.0, 0.9, 0.8, 0.95, 0.85], 0.92)
    ]
    data.extend(examples)
    return data


def score_resume(resume):
    features = extract_features(resume)
    network = NeuralNetwork(len(features))
    network.train(make_training_data())
    raw_score = network.predict(features)
    score = max(1, min(10, round(raw_score * 9 + 1)))
    return score, features


def feedback(features):
    labels = [
        "Resume length",
        "Technical skills",
        "Action verbs",
        "Education",
        "Important sections",
        "Contact details",
        "Measurable results",
        "Bullet structure",
        "Readable lines",
        "Vocabulary range"
    ]
    strengths = [label for label, value in zip(labels, features) if value >= 0.65]
    improvements = [label for label, value in zip(labels, features) if value < 0.45]
    return strengths[:4], improvements[:4]


def read_resume_from_file(path):
    file_path = Path(path)
    if not file_path.exists():
        print("File not found.")
        return ""
    if file_path.suffix.lower() != ".txt":
        print("Only .txt files are supported.")
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


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
    source = input("Enter txt file path or press Enter to paste resume: ").strip().strip('"')
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
