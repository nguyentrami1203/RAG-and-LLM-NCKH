"""Wrapper mỏng gọi LLM. Model spec dạng  provider:model  (anthropic | openai).
openai cũng dùng được cho Gemini/DeepSeek/Llama qua endpoint tương thích: đặt OPENAI_BASE_URL + OPENAI_API_KEY."""


def chat(spec, system, messages, max_tokens=700):
    provider, model = spec.split(":", 1)
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
        return "".join(b.text for b in r.content if b.type == "text")
    if provider == "openai":
        from openai import OpenAI
        client, msgs = OpenAI(), [{"role": "system", "content": system}] + messages
        try:
            r = client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=max_tokens)
        except Exception:
            r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
        return r.choices[0].message.content
    raise ValueError(f"unknown provider {provider}")
