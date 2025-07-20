import google.generativeai as genai

genai.configure(api_key="AIzaSyBgJQT77kkCXRFa74CUCPUZpfjlQQsZQao")



def suggest_titles(original_title):
    prompt = f"""
هذا عنوان لمقال: \"{original_title}\"
اقترح 3 عناوين بديلة أكثر جذبًا ووضوحًا، يجب أن تكون باللغة العربية ومناسبة لجمهور من المدراء والموظفين المهتمين بتطوير الذات، ولا تتجاوز 15 كلمة.
اكتب كل عنوان في سطر منفصل فقط بدون شروحات.
"""
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    suggestions = response.text.strip().split("\n")
    return [s.strip("•- ") for s in suggestions if s.strip()]
