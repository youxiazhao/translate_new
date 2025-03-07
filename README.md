test_texts.json 文件示例：

```json
[
  {
    "text": "Ghola galaat hai",
    "target_language": "中文",
    "description": "之前出错的印第安语1"
  },
  {
    "text": "Vhi na meko to day first se pta tha",
    "target_language": "中文",
    "description": "之前出错的印第安语2"
  },
  {
    "text": "Bhejna nhi chalna hai ais bolo",
    "target_language": "中文",
    "description": "之前出错的印第安语3"
  }
]
```

model_combinations.json 文件示例：

```json
{
  "combinations": [
    {
      "name": "全部使用 GPT 4o mini",
      "translator": ["openai", "gpt-4o-mini"],
      "evaluator": ["openai", "gpt-4o-mini"],
      "retry_translator": ["openai", "gpt-4o-mini"]
    },
    {
      "name": "全部使用 Gemini 2.0 Flash",
      "translator": ["google", "gemini-2.0-flash"],
      "evaluator": ["google", "gemini-2.0-flash"],
      "retry_translator": ["google", "gemini-2.0-flash"]
    },
    {
      "name": "全部使用 Gemini 2.0 Flash Lite",
      "translator": ["google", "gemini-2.0-flash-lite"],
      "evaluator": ["google", "gemini-2.0-flash-lite"],
      "retry_translator": ["google", "gemini-2.0-flash-lite"]
    },
    {
      "name": "lite翻译 + flash评估 + gpt重试",
      "translator": ["google", "gemini-2.0-flash-lite"],
      "evaluator": ["google", "gemini-2.0-flash"],
      "retry_translator": ["openai", "gpt-4o-mini"]
    },
    {
      "name": "flash翻译 + lite评估 + gpt重试",
      "translator": ["google", "gemini-2.0-flash"],
      "evaluator": ["google", "gemini-2.0-flash-lite"],
      "retry_translator": ["openai", "gpt-4o-mini"]
    }
  ]
}
```
