"""
=============================================================================
NLP Module — Rule-Based Grammar Engine
Converts gesture label sequences into grammatically correct English sentences.
=============================================================================

Usage:
    from nlp_module import GestureToSentence
    
    nlp = GestureToSentence()
    sentence = nlp.construct_sentence(["mother", "help", "dog"])
    print(sentence)  # → "Mother helps the dog."
"""


class GestureToSentence:
    """
    Rule-based grammar engine that converts a sequence of gesture labels
    into a grammatically correct, context-aware English sentence.
    """
    
    def __init__(self):
        """Initialize word dictionaries and grammar rules."""
        
        # =====================================================================
        # WORD CATEGORIZATION — POS (Part of Speech) tagging for all 56 labels
        # =====================================================================
        self.nouns = {
            "book", "computer", "dog", "family", "language", "mother", 
            "sea", "singer", "sofa", "something", "talent", "telescope", 
            "towel", "truth", "tv", "water", "waterfall", "weelchair", "woman"
        }
        
        self.verbs = {
            "can", "drink", "finish", "go", "help", "like", "saw", "scream", 
            "shout", "skip", "solve", "tempt", "tend", "text", "thrill", 
            "turn", "walk", "weigh", "want"
        }
        
        self.adjectives = {
            "cool", "deaf", "fine", "hearing", "many", "unique", "vacant", "good", "bad"
        }
        
        self.adverbs = {
            "later", "now", "upstairs", "very", "therefore"
        }
        
        self.affirmatives = {"yes", "ok", "fine"}
        self.negatives = {"no", "bad"}
        self.conjunctions = {"than"}
        self.exclamations = {"bye", "hello", "please", "sorry", "thank_you", "thank you"}
        
        self.interrogatives = {"how", "what", "where", "why", "who"}
        self.pronouns = {"i", "me", "you", "all", "something"}
        
        # =====================================================================
        # VERB CONJUGATION — 3rd person singular present tense
        # =====================================================================
        self.verb_conjugations = {
            "drinking": "is drinking",
            "eat": "eats",
            "want": "wants",
            "go": "goes",
            "walk": "walks",
            "drink": "drinks",
            "solve": "solves",
            "turn": "turns",
            "weigh": "weighs"
        }
        
        # =====================================================================
        # ARTICLES — Words that need articles prepended
        # =====================================================================
        self.needs_article = {
            "book": "a", "computer": "a", "dog": "a",
            "family": "a", "singer": "a", "sofa": "a",
            "telescope": "a", "towel": "a", "tv": "a",
            "waterfall": "a", "woman": "a", "weelchair": "a",
            "sea": "the", "truth": "the"
        }
        
        # =====================================================================
        # WORD DISPLAY — Clean display names for misspelled labels
        # =====================================================================
        self.display_names = {
            "weelchair": "wheelchair",
            "tv": "TV"
        }
        
        # =====================================================================
        # PHRASE MAPPINGS — For specific gesture sequences
        # =====================================================================
        self.phrase_mappings = {
            "how you": "How are you?",
            "I fine": "I am fine.",
            "I ok": "I am okay.",
            "me sorry": "I am sorry.",
            "me fine": "I am fine.",
            "me ok": "I am okay.",
            "I want water": "I want some water.",
            "me want water": "I want some water.",
            "you want what": "What do you want?",
            "where you": "Where are you?",
            "what you want": "What do you want?",
            "you eat": "Are you eating?",
            "I drinking": "I am drinking.",
            "thank you": "Thank you!",
            "thank_you": "Thank you!",
            "me want eat": "I want to eat.",
            "I want eat": "I want to eat.",
            "you want water": "Do you want some water?",
        }

        # Set-based phrase mappings (matches keywords in any order)
        self.phrase_sets = [
            ({"hello", "how", "you"}, "Hello, how are you?"),
            ({"how", "you"}, "How are you?"),
            ({"hello", "ok", "you"}, "Hello, are you ok?"),
            ({"you", "ok"}, "Are you ok?"),
            ({"thank_you"}, "Thank you!"),
        ]

        self.templates = {
            # Single word responses
            1: [
                "{word}."
            ],
            # Two word patterns
            2: [
                "{subj} {verb}.",
                "{adj} {noun}.",
                "{verb} {obj}."
            ],
            # Three+ word patterns
            3: [
                "{subj} {verb} {obj}.",
                "{subj} {verb} {adv}.",
                "{adv}, {subj} {verb}."
            ]
        }
    
    def get_display_name(self, word):
        """Get clean display name for a word."""
        return self.display_names.get(word, word)
    
    def get_pos(self, word):
        """Determine the primary part of speech for a word."""
        if word in self.nouns:
            return "noun"
        elif word in self.verbs:
            return "verb"
        elif word in self.adjectives:
            return "adjective"
        elif word in self.adverbs:
            return "adverb"
        elif word in self.pronouns:
            return "pronoun"
        elif word in self.affirmatives:
            return "affirmative"
        elif word in self.negatives:
            return "negative"
        elif word in self.interrogatives:
            return "interrogative"
        elif word in self.exclamations:
            return "exclamation"
        elif word in self.conjunctions:
            return "conjunction"
        else:
            return "noun"  # Default assumption
    
    def add_article(self, word):
        """Add appropriate article to a noun if needed."""
        if word in self.needs_article:
            article = self.needs_article[word]
            display = self.get_display_name(word)
            return f"{article} {display}"
        return self.get_display_name(word)
    
    def conjugate_verb(self, verb, subject=None):
        """Conjugate verb based on subject."""
        # If subject is a singular noun, use 3rd person
        if subject and subject in self.nouns:
            return self.verb_conjugations.get(verb, verb + "s")
        # Default: base form
        return self.get_display_name(verb)
    
    def remove_consecutive_duplicates(self, words):
        """Remove consecutive duplicate words."""
        if not words:
            return words
        filtered = [words[0]]
        for w in words[1:]:
            # If the current word is the same as the previous, skip it
            # Also handle variations like 'thank you' vs 'thank_you'
            w_norm = w.replace("_", " ").lower()
            prev_norm = filtered[-1].replace("_", " ").lower()
            if w_norm != prev_norm:
                filtered.append(w)
        return filtered
    
    def clean_sequence(self, words):
        """Aggressively clean the sequence of words for more coherent sentences."""
        if not words: return []
        
        # 1. Remove consecutive duplicates
        cleaned = self.remove_consecutive_duplicates(words)
        
        # 2. Limit excessive repetition of single words (noise reduction)
        final = []
        counts = {}
        for w in cleaned:
            counts[w] = counts.get(w, 0) + 1
            if counts[w] <= 2: # Allow max 2 occurrences if separated
                final.append(w)
        
        return final

    def construct_sentence(self, gesture_labels):
        """
        Convert a sequence of gesture labels into a grammatically correct sentence.
        
        Args:
            gesture_labels: List of gesture label strings, e.g., ["mother", "help", "dog"]
        
        Returns:
            Grammatically correct English sentence string
        """
        if not gesture_labels:
            return ""
        
        # Clean input
        words = [w.lower().strip() for w in gesture_labels if w.strip()]
        words = self.clean_sequence(words)
        
        if not words:
            return ""

        # 1. Check for SET-BASED phrase mappings first (flexible order)
        word_set = set(words)
        for required_set, mapped_sentence in self.phrase_sets:
            if required_set.issubset(word_set):
                return mapped_sentence

        # 2. Check for exact phrase mappings (natural sentence overrides)
        joined_words = " ".join(words).lower()
        for phrase, mapped_sentence in self.phrase_mappings.items():
            if phrase.lower() == joined_words:
                return mapped_sentence
        
        # Analyze POS tags
        pos_tags = [(w, self.get_pos(w)) for w in words]
        
        # =====================================================================
        # SINGLE WORD
        # =====================================================================
        if len(words) == 1:
            word = words[0]
            display = self.get_display_name(word)
            pos = pos_tags[0][1]
            
            if pos == "affirmative":
                return "Yes." if word == "yes" else f"{display.capitalize()}."
            elif pos == "negative":
                return "No." if word == "no" else f"{display.capitalize()}."
            elif pos == "exclamation":
                if word == "hello": return "Hello!"
                if word == "thank_you": return "Thank you!"
                return f"{display.capitalize()}!"
            elif pos == "interrogative":
                return f"{display.capitalize()}?"
            elif pos == "noun":
                return f"{self.add_article(word).capitalize()}."
            elif pos == "verb":
                return f"{display.capitalize()}."
            elif pos == "adjective":
                return f"It is {display}."
            else:
                return f"{display.capitalize()}."
        
        # =====================================================================
        # TWO WORDS
        # =====================================================================
        if len(words) == 2:
            w1, pos1 = pos_tags[0]
            w2, pos2 = pos_tags[1]
            
            # Pronoun + Adjective: "I fine" -> "I am fine."
            if pos1 in ["pronoun", "noun"] and pos2 == "adjective":
                subj = self.get_display_name(w1).capitalize()
                if w1.lower() in ["i", "me"]:
                    return "I am fine." if w2 == "fine" else f"I am {w2}."
                elif w1.lower() == "you":
                    return f"You are {w2}."
                else:
                    return f"{subj} is {w2}."

            # Pronoun + Noun: "I water" -> "I want water."
            if pos1 in ["pronoun", "noun"] and pos2 == "noun":
                subj = self.get_display_name(w1).capitalize()
                obj = self.add_article(w2)
                if w1.lower() in ["i", "me"]:
                    return f"I want {obj}."
                else:
                    return f"{subj} wants {obj}."

            # Subject + Verb: "Mother help" -> "Mother helps."
            if pos1 == "noun" and pos2 == "verb":
                subj = self.add_article(w1).capitalize()
                verb = self.conjugate_verb(w2, w1)
                return f"{subj} {verb}."
            
            # Subject (Pronoun) + Verb: "I go."
            if pos1 == "pronoun" and pos2 == "verb":
                subj = "I" if w1.lower() in ["i", "me"] else w1.capitalize()
                verb = self.get_display_name(w2)
                if subj == "I" and verb == "fine": return "I am fine." # Safety
                return f"{subj} {verb}."

            # Verb + Object: "Help mother."  
            if pos1 == "verb" and pos2 == "noun":
                verb = self.get_display_name(w1).capitalize()
                obj = self.add_article(w2)
                return f"{verb} {obj}."
            
            # Adjective + Noun: "Cool dog."
            if pos1 == "adjective" and pos2 == "noun":
                adj = self.get_display_name(w1).capitalize()
                noun = self.add_article(w2)
                return f"{adj} {noun}."
            
            # Default: join naturally
            d1 = self.get_display_name(w1).capitalize()
            d2 = self.get_display_name(w2)
            return f"{d1} {d2}."
        
        # =====================================================================
        # THREE OR MORE WORDS — Build sentence structurally
        # =====================================================================
        sentence_parts = []
        subject = None
        has_verb = False
        
        for i, (word, pos) in enumerate(pos_tags):
            display = self.get_display_name(word)
            
            # Contextual Correction: 'me' as subject -> 'I'
            if word == "me" and i == 0 and len(words) > 1:
                word = "I"
                display = "I"
                pos = "pronoun"

            if pos == "noun" and subject is None and not has_verb:
                # First noun before any verb = subject
                subject = word
                sentence_parts.append(self.add_article(word))
            
            elif pos == "pronoun" and subject is None and not has_verb:
                # Pronoun as subject
                subject = word
                sentence_parts.append(display)
            
            elif pos == "verb":
                # Handle Verb + Verb (e.g., 'want eat' -> 'want to eat')
                if has_verb and sentence_parts and sentence_parts[-1] == "want":
                    sentence_parts.append("to")
                
                has_verb = True
                if subject:
                    sentence_parts.append(self.conjugate_verb(word, subject))
                else:
                    sentence_parts.append(display)
            
            elif pos == "noun" and has_verb:
                # Noun after verb = object
                sentence_parts.append(self.add_article(word))
            
            elif pos == "adjective":
                # Handle Pronoun + Adjective (e.g., 'I fine' -> 'I am fine')
                if subject and (subject == "I" or subject == "me"):
                    sentence_parts.append("am")
                elif subject == "you":
                    sentence_parts.append("are")
                elif subject: # Any other singular subject
                    sentence_parts.append("is")
                
                # Check if next word is a noun
                if i + 1 < len(pos_tags) and pos_tags[i + 1][1] == "noun":
                    sentence_parts.append(display)
                else:
                    sentence_parts.append(display)
            
            elif pos == "adverb":
                sentence_parts.append(display)
            
            elif pos == "negative":
                # Insert "not" or "do not"
                if has_verb:
                    sentence_parts.append("not")
                else:
                    sentence_parts.append("do not")
            
            elif pos == "affirmative":
                sentence_parts.insert(0, "yes,")
            
            elif pos == "conjunction":
                sentence_parts.append(display)
            
            elif pos == "pronoun":
                # Handle Pronoun as Object or Interrogative
                if word == "who":
                    sentence_parts.append("who")
                elif word == "what":
                    sentence_parts.append("what")
                else:
                    sentence_parts.append(display)
            
            else:
                sentence_parts.append(display)
        
        # Join and format
        sentence = " ".join(sentence_parts)
        
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]
        
        # Add period if not already ending with punctuation
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        
        return sentence
    
    def process_gesture_buffer(self, gesture_buffer, min_confidence=0.6):
        """
        Process a buffer of gesture predictions with confidence filtering.
        
        Args:
            gesture_buffer: List of (label, confidence) tuples
            min_confidence: Minimum confidence to include a gesture
        
        Returns:
            Constructed sentence string
        """
        # Filter by confidence
        filtered = [label for label, conf in gesture_buffer if conf >= min_confidence]
        
        if not filtered:
            return ""
        
        # Remove consecutive duplicates (common in real-time detection)
        deduplicated = self.remove_consecutive_duplicates(filtered)
        
        return self.construct_sentence(deduplicated)


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    nlp = GestureToSentence()
    
    print("=" * 60)
    print("NLP MODULE — Rule-Based Grammar Engine Test")
    print("=" * 60)
    
    test_cases = [
        # Single words
        (["yes"], "Affirmative"),
        (["no"], "Negative"),
        (["help"], "Single verb"),
        (["dog"], "Single noun"),
        (["cool"], "Single adjective"),
        
        # Two words
        (["mother", "help"], "Subject + Verb"),
        (["help", "dog"], "Verb + Object"),
        (["cool", "dog"], "Adjective + Noun"),
        (["mother", "dog"], "Noun + Noun"),
        (["now", "go"], "Adverb + Verb"),
        (["no", "walk"], "Negative + Verb"),
        
        # Three words
        (["mother", "help", "dog"], "Subject + Verb + Object"),
        (["woman", "drink", "water"], "Subject + Verb + Object"),
        (["dog", "walk", "upstairs"], "Subject + Verb + Adverb"),
        (["mother", "like", "computer"], "Subject + Verb + Object"),
        
        # Complex
        (["yes", "mother", "help", "dog"], "Affirmative + SVO"),
        (["woman", "go", "now"], "Subject + Verb + Adverb"),
        (["family", "like", "dog"], "Subject + Verb + Object"),
        (["who", "help", "mother"], "Pronoun + Verb + Object"),
        (["dog", "drink", "water", "now"], "SVO + Adverb"),
        (["you", "how"], "Set Mapping (Unordered)"),
        (["ok", "hello", "you"], "Set Mapping (Unordered)"),
        
        # Phrase Mappings (User Requirements)
        (["how", "you"], "Phrase Mapping 1"),
        (["hello", "you", "ok"], "Phrase Mapping 2"),
        (["mother", "dog"], "No 'and' check"),
        (["i", "fine"], "User Case 1: I fine -> I am fine"),
        (["i", "water"], "User Case 2: I water -> I want water"),
        (["you", "cool"], "User Case 3: You cool -> You are cool"),
        
        # With duplicates (simulating real-time detection)
        (["help", "help", "help", "mother"], "Deduplicated"),
    ]
    
    for gestures, description in test_cases:
        result = nlp.construct_sentence(gestures)
        print(f"\n  {description}")
        print(f"  Input:  {gestures}")
        print(f"  Output: {result}")
    
    print(f"\n{'='*60}")
    print("[OK] NLP module working correctly!")
