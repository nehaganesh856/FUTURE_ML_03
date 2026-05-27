import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")

def clean_and_parse(text):
    """Clean and normalize text for analysis."""
    doc = nlp(text.lower())
    
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_skills(text, skill_db):
    """Identify matching skills and gaps."""
    text_set = set(text.lower().split())
    found_skills = text_set.intersection(skill_db)
    missing_skills = skill_db.difference(found_skills)
    return found_skills, missing_skills


job_description = "Looking for a Data Scientist proficient in Python, Scikit-learn, and NLP."
resumes = [
    "Experienced developer with skills in Python, Java, and Web Development.",
    "Data Scientist expert in Python, Scikit-learn, and Natural Language Processing (NLP).",
    "Project manager with a focus on Agile and Scrum."
]
required_skills = {"python", "scikit-learn", "nlp"}


cleaned_jd = clean_and_parse(job_description)
cleaned_resumes = [clean_and_parse(r) for r in resumes]


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([cleaned_jd] + cleaned_resumes)
scores = cosine_similarity(vectors[0:1], vectors[1:])[0]


print("--- Candidate Ranking Results ---")
for i, score in enumerate(scores):
    skills, gaps = extract_skills(resumes[i], required_skills)
    print(f"\nCandidate {i+1}:")
    print(f"Match Score: {score:.2f}") 
    print(f"Skills Found: {', '.join(skills) if skills else 'None'}")
    print(f"Skill Gaps: {', '.join(gaps) if gaps else 'None'}") 