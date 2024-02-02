import nltk
conversation = [
    "Hello, how are you?",
    "I'm doing well, thank you.",
    "Can you help me with my homework?",
    "Sure, I'd be happy to help.",
    "Great! I have some math problems to solve."
]
def recognize_dialog_acts(sentences):
    dialog_acts = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        parsed = nltk.ne_chunk(tagged)
        dialog_act = None
        verbs = [word for word, pos in tagged if pos.startswith('V')]
        modal_verbs = [word for word, pos in tagged if pos == 'MD']
        keywords = ["help", "homework", "problem", "solve"]
        if "can" in modal_verbs and any(keyword in sentence for keyword in keywords):
            dialog_act = "Request for Help"
        elif any(verb in verbs for verb in ["help", "assist", "support"]) and "you" in tokens:
            dialog_act = "Offer Help"
        elif any(verb in verbs for verb in ["solve", "answer"]) and any(keyword in sentence for keyword in keywords):
            dialog_act = "Task Acknowledgment"
        else:
            dialog_act = "Other"
        dialog_acts.append((sentence, dialog_act))
    return dialog_acts
dialog_acts = recognize_dialog_acts(conversation)
for sentence, act in dialog_acts:
    print(f"Dialog Act: {act}\nSentence: {sentence}\n")
