# Translation Prompt Template
TRANSLATION_PROMPT_TEMPLATE = """You are a professional translator. Please carefully follow these guidelines:

1. If the text doesn't make sense in any language, such as containing only special characters, return it as is.
2. Detect the language of the following text: {text} as source language.
3. If the detected source language is the same as the target language {target_language}:
   - Return the original text as is
4. If the detected source language differs from the target language:
   - Translate {text} into {target_language}
   - Preserve all formatting, numbers, and special characters
5. Special handling for multilingual text:
   - If the text contains multiple languages, identify the primary language
   - For mixed language texts (like Hinglish, Spanglish, etc.), translate the meaning correctly rather than just transliterating
   - When translating latinized text that may represent non-Latin script languages (e.g., "Puch rhi hu" instead of "पूछ रही हूँ"), translate the meaning into the target language rather than just converting to the original script
   - For code-switching text (alternating between languages), translate the overall meaning while preserving technical terms
6. For transliterated text:
   - Focus on translating the meaning into {target_language}, not converting to the original script
   - Example: If input is "Main theek hun" and target is Chinese, output should be "我很好" not "मैं ठीक हूँ"

Please provide the translation, source language, and a brief explanation of why your translation is correct.

Original text: {text}

response_schema {{
  "source_language": "string",
  "translated_text": "string",
  "explanation": "string"
}}"""

# Evaluation Prompt Template
EVALUATION_PROMPT_TEMPLATE = """Please evaluate the quality of the following translation by scoring each dimension separately and calculating the total score.

Input parameters:
- target_language: target language
- original_text: original text
- translated_text: translated text

Original text: {original_text}

Translation ({target_language}): {translated_text}

Please evaluate the translation quality based on the following five dimensions, with each dimension scored out of 10:
1. Language accuracy: whether the translation is in the correct target language; if not, this item directly scores 0, and please indicate this in the feedback
2. Semantic accuracy: whether the translation accurately conveys the meaning of the original text
3. Fluency: whether the translation reads naturally and smoothly
4. Style: whether the translation maintains the style and tone of the original text
5. Terminology: if there are specialized terms, whether they are correctly translated

Please score each dimension separately, then calculate the final score (total score = sum of the five dimension scores / 5).

response_schema {{
  "scores": {{
    "language_accuracy": "number",
    "semantic_accuracy": "number",
    "fluency": "number",
    "style": "number",
    "terminology": "number"
  }},
  "total_score": "number",
  "feedback": "string"
}}"""

# Retry Translation Prompt Template
RETRY_TRANSLATION_PROMPT_TEMPLATE = """Please improve the translation of the following text from its original language to {target_language} based on the previous evaluation feedback.

Original text: {text}

Previous translation: {previous_translation}

Previous translation evaluation feedback:
Total score: {feedback.total_score}
Dimension scores:
- Language accuracy: {feedback.scores.language_accuracy}
- Semantic accuracy: {feedback.scores.semantic_accuracy}
- Fluency: {feedback.scores.fluency}
- Style: {feedback.scores.style}
- Terminology accuracy: {feedback.scores.terminology}
Detailed feedback: {feedback.feedback}

Please focus on improving dimensions with lower scores based on the feedback above, ensuring the translation is accurate, fluent, natural, and maintains the style and tone of the original text."""

# 检查 RETRY_TRANSLATION_PROMPT_TEMPLATE 的内容
print(RETRY_TRANSLATION_PROMPT_TEMPLATE)

# 定义统一的 JSON Schema (用于 OpenAI API)
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "target_language": {
            "type": "string",
            "description": "目标语言"
        },
        "translated_text": {
            "type": "string",
            "description": "翻译后的文本"
        }
    },
    "required": ["target_language", "translated_text"]
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "language_accuracy": {
                    "type": "number",
                    "description": "语言准确性评分 (0-10)"
                },
                "semantic_accuracy": {
                    "type": "number",
                    "description": "语义准确性评分 (0-10)"
                },
                "fluency": {
                    "type": "number",
                    "description": "流畅性评分 (0-10)"
                },
                "style": {
                    "type": "number",
                    "description": "风格评分 (0-10)"
                },
                "terminology": {
                    "type": "number",
                    "description": "术语准确性评分 (0-10)"
                }
            },
            "required": ["language_accuracy", "semantic_accuracy", "fluency", "style", "terminology"],
            "description": "各个维度的评分"
        },
        "total_score": {
            "type": "number",
            "description": "总评分 (0-10，为五个维度分数的平均值)"
        },
        "feedback": {
            "type": "string",
            "description": "详细的评估反馈"
        }
    },
    "required": ["scores", "total_score", "feedback"]
}

# 重试翻译的 JSON Schema (用于 OpenAI API)
RETRY_TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "target_language": {
            "type": "string",
            "description": "目标语言"
        },
        "translated_text": {
            "type": "string",
            "description": "改进后的翻译文本"
        }
    },
    "required": ["target_language", "translated_text"]
}

# 获取对应的 schema
def get_schema_for_prompt_type(prompt_type):
    """根据提示类型返回相应的 schema"""
    schema_map = {
        "translation": TRANSLATION_SCHEMA,
        "evaluation": EVALUATION_SCHEMA,
        "retry_translation": RETRY_TRANSLATION_SCHEMA,
    }
    return schema_map.get(prompt_type, {})