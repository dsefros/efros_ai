def step1(ctx, kernel):
    text = ctx["payload"]["text"]
    return {"text": text.upper()}

def step2(ctx, kernel):
    return {"result": ctx["text"] + "!!!"}

PIPELINE = [
    step1,
    step2
]
