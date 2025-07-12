from transformers import pipeline

# Load model
quiz_gen = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_quiz(documents):
    context = " ".join([doc.page_content for doc in documents[:3]])  # Take top 3 chunks
    prompt = f"""
You are a quiz generator.

Create 5 multiple-choice questions (MCQs) with 4 options and indicate the correct answer clearly like:
**Answer: A**

Content:
{context[:1000]}
"""
    output = quiz_gen(prompt, max_new_tokens=256)[0]["generated_text"]
    return output

def format_quiz(raw_text):
    # Naive formatter: split by lines and group
    questions = []
    current = {}
    lines = raw_text.split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("q") and "?" in line:
            if current:
                questions.append(current)
            current = {"question": line, "options": [], "answer": ""}
        elif line.startswith("A.") or line.startswith("B.") or line.startswith("C.") or line.startswith("D."):
            current["options"].append(line)
        elif "Answer:" in line:
            current["answer"] = line.split("Answer:")[-1].strip()
    if current:
        questions.append(current)
    return questions
