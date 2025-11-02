import spacy
import re

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize the text."""
    doc = nlp(text.lower())

    # Keep tokens that are not stop words or punctuation. This is safer than is_alpha.
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Join the tokens and handle the case where the list might be empty
    processed_text = " ".join(clean_tokens).strip()
    
    # If the text is empty after processing, return a unique non-matching string
    return processed_text if processed_text else "EMPTY_PROCESSED_STRING"

def match_keywords(resume_text: str, keywords: list[str]) -> tuple[list[str], str]:
    """Match keywords in the resume to the job description.
    
    Returns:
        tuple containing (matched_keywords, decision)
    """
    # Preprocess both the resume and keywords
    processed_resume = preprocess_text(resume_text)

    keywords_found = []
    for original_keyword in keywords:
        
        if len(original_keyword.split()) > 1:  # if it's a phrase like "software development"
            processed_keyword =  preprocess_text(original_keyword)  # spaCy-based cleanup
        else:
            processed_keyword =  original_keyword.lower().strip()  # preserve atomic terms like OAuth2, .NET, etc.
        
        # Skip if the processed keyword is empty
        if not processed_keyword or processed_keyword == "EMPTY_PROCESSED_STRING":
            continue

        pattern = r'\b' + re.escape(processed_keyword) + r'\b'
        if re.search(pattern, processed_resume):
            keywords_found.append(original_keyword)
    
    print(f"Keywords found: {keywords_found}")

    # Make preliminary decision based on simple logic
    if len(keywords_found) == 0:
        initial_decision = "FAIL - No keywords matched"
    elif len(keywords_found) < len(keywords) * 0.3:  # Less than 30% match
        initial_decision = "WEAK - Low keyword match rate"
    elif len(keywords_found) >= len(keywords) * 0.6:  # 60% or more match
        initial_decision = "STRONG - High keyword match rate"
    else:  # Between 30% and 60%
        initial_decision = "MODERATE - Moderate keyword match rate"
    
    return keywords_found, initial_decision
