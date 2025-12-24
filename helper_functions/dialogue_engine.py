from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import re
import time
import logging
from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)

class ConversationPhase(Enum):
    GREETING = "greeting"
    MAIN = "main"
    CLOSING = "closing"

@dataclass
class DialogueTurn:
    character_id: str
    text: str
    timestamp: float
    token_count: int = 0
    estimated_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "estimated_duration": self.estimated_duration
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTurn':
        return cls(
            character_id=data["character_id"],
            text=data["text"],
            timestamp=data["timestamp"],
            token_count=data.get("token_count", 0),
            estimated_duration=data.get("estimated_duration", 0.0)
        )

class DialogueEngine:
    """
    Core component responsible for simulating natural, multi-turn conversations 
    between two AI personas.
    """
    
    def __init__(self, config: Any, use_chatterbox: bool = False):
        self.config = config
        self.use_chatterbox = use_chatterbox
        self.turns: List[DialogueTurn] = []
        self.character_a = config.podcast_voice_a
        self.character_b = config.podcast_voice_b
        self.turn_order = [self.character_a, self.character_b]
        
        # Heuristic for duration estimation: 150 words per minute
        self.wpm = 150
        self.words_per_second = self.wpm / 60.0
        
    def run_conversation(self, topic: str, context: str) -> List[DialogueTurn]:
        """
        Main entry point to orchestrate the conversation loop.
        """
        self.turns = []
        target_duration = self.config.podcast_target_duration_seconds
        max_turns = self.config.podcast_max_turns
        
        print(f"🎙️ Starting podcast dialogue between {self.character_a} and {self.character_b}...")
        
        while len(self.turns) < max_turns:
            # Check stopping conditions
            total_estimated_duration = sum(t.estimated_duration for t in self.turns)
            if total_estimated_duration >= target_duration:
                print(f"  Reached target duration ({total_estimated_duration:.1f}s / {target_duration}s)")
                break
                
            # Execute one turn
            try:
                turn = self.execute_turn(topic, context)
                if turn:
                    self.turns.append(turn)
                    print(f"  Turn {len(self.turns)}: [{turn.character_id}] {turn.text[:50]}...")
                else:
                    break
            except Exception as e:
                logger.error(f"Error in dialogue turn: {e}")
                break
                
        return self.turns

    def execute_turn(self, topic: str, context: str) -> Optional[DialogueTurn]:
        """
        Executes a single dialogue turn.
        """
        character_id = self._get_active_character()
        phase = self._get_current_phase()
        
        # Determine LLM config based on character
        if character_id == self.character_a:
            provider = self.config.podcast_voice_a_provider
            model_name = self.config.podcast_voice_a_model_name
        else:
            provider = self.config.podcast_voice_b_provider
            model_name = self.config.podcast_voice_b_model_name

        # Call LLM using configured provider and model name
        client = get_client(provider=provider, model_name=model_name)
        
        # Build conversation history for the rolling window
        history = self._get_rolling_window_history()
        
        messages = [
            {"role": "system", "content": self._get_system_message(character_id, topic, context)},
        ]
        
        # Add history as user/assistant turns
        # We need to be careful with character mapping. 
        # For character_id, the 'assistant' role is themselves.
        for t in history:
            role = "assistant" if t.character_id == character_id else "user"
            content = f"{t.character_id}: {t.text}" if role == "user" else t.text
            messages.append({"role": role, "content": content})
            
        # Add current instruction
        messages.append({"role": "user", "content": self._get_phase_instruction(character_id, phase)})
        
        response = client.chat_completion(
            messages=messages,
            temperature=0.8, # Higher temperature for more natural dialogue
            max_tokens=500
        )
        
        cleaned_text = self._clean_response(response, character_id)
        
        if not cleaned_text:
            return None
            
        # Estimate duration
        word_count = len(cleaned_text.split())
        duration = word_count / self.words_per_second
        
        return DialogueTurn(
            character_id=character_id,
            text=cleaned_text,
            timestamp=time.time(),
            estimated_duration=duration
        )

    def _get_active_character(self) -> str:
        """Determines who speaks next based on alternating order."""
        num_turns = len(self.turns)
        return self.turn_order[num_turns % len(self.turn_order)]

    def _get_current_phase(self) -> ConversationPhase:
        """Determines the current conversation phase."""
        num_turns = len(self.turns)
        if num_turns < 2:
            return ConversationPhase.GREETING
            
        total_estimated_duration = sum(t.estimated_duration for t in self.turns)
        target_duration = self.config.podcast_target_duration_seconds
        
        # Last 30 seconds or last 2 turns trigger closing
        if total_estimated_duration >= (target_duration - 30) or num_turns >= (self.config.podcast_max_turns - 2):
            return ConversationPhase.CLOSING
            
        return ConversationPhase.MAIN

    def _get_system_message(self, character_id: str, topic: str, context: str) -> str:
        co_host = self.character_b if character_id == self.character_a else self.character_a
        
        # Chatterbox TTS paralinguistic tag instructions for engaging podcast delivery
        chatterbox_instruction = ""
        if self.use_chatterbox:
            chatterbox_instruction = """
6. PARALINGUISTIC EXPRESSION TAGS - Make your delivery engaging and human:
   - [laugh] - use when reacting to absurd news, surprising stats, or genuine humor
   - [chuckle] - use for witty observations, ironic situations, or playful banter with co-host
   - [cough] - use before dropping a bombshell fact or pivoting to something serious
   
   BE EXPRESSIVE! This is a podcast, not a boring news read. Use 2-3 tags per response.
   React naturally to what your co-host says and to surprising news.
   Examples:
   - "Wait, wait, wait [laugh] did you just say a TRILLION dollars?"
   - "[chuckle] Oh man, the irony is just too good here."
   - "Alright [cough] but here's where it gets really interesting..."
   - "You know what [laugh] I actually saw that coming!"
"""
        
        return f"""You are {character_id.replace('_', ' ').title()}, a distinct personality.
You are co-hosting a podcast with {co_host.replace('_', ' ').title()}.
The topic today is: {topic}
Research Context: {context}

IMPORTANT: You must ensure all key points from the Research Context are discussed before the podcast ends.

RULES:
1. Stay strictly in character. Use your signature vocabulary, rhythm, and style.
2. Be conversational and responsive to your co-host.
3. Keep your responses concise (max {self.config.podcast_max_sentences_per_turn} sentences).
4. Do NOT write dialogue for your co-host. Speak only for yourself.
5. Do NOT use speaker labels like '{character_id.title()}:'.
{chatterbox_instruction}"""

    def _get_phase_instruction(self, character_id: str, phase: ConversationPhase) -> str:
        co_host = self.character_b if character_id == self.character_a else self.character_a
        co_host_name = co_host.replace('_', ' ').title()
        
        if phase == ConversationPhase.GREETING:
            if len(self.turns) == 0:
                return f"Start the podcast with a casual, friendly greeting to your co-host {co_host_name}. Mention you're excited to discuss the news today."
            else:
                return f"Respond naturally to {co_host_name}'s greeting and get ready to dive into the news."
        elif phase == ConversationPhase.CLOSING:
            return "The conversation is wrapping up now. Offer some final thoughts on what you discussed today, then say a casual goodbye to your co-host and the listeners."
        else:
            return f"Continue the discussion about the news briefing. Respond to what {co_host_name} just said and add your own perspective based on the research context."

    def _get_rolling_window_history(self) -> List[DialogueTurn]:
        """Returns the last N turns for context."""
        window_size = self.config.podcast_context_window_turns
        return self.turns[-window_size:] if len(self.turns) > window_size else self.turns

    def _clean_response(self, text: str, character_id: str) -> str:
        """Sanitizes LLM output."""
        # 1. Remove speaker labels like "Rick:" or "Rick Sanchez:"
        name_part = character_id.replace('_', ' ').title()
        id_part = character_id.title()
        patterns = [
            f"^{name_part}:", f"^{id_part}:", 
            f"^{character_id}:", f"^{character_id.lower()}:"
        ]
        cleaned = text.strip()
        for p in patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
            
        # 2. Leakage Truncation: remove if it starts writing for the other character
        co_host = self.character_b if character_id == self.character_a else self.character_a
        co_host_name = co_host.replace('_', ' ').title()
        co_host_id = co_host.title()
        
        leakage_patterns = [
            f"\n{co_host_name}:", f"\n{co_host_id}:", f"\n{co_host}:"
        ]
        for p in leakage_patterns:
            if p in cleaned:
                cleaned = cleaned.split(p)[0].strip()
                
        # 3. Sentence Capping
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        max_s = self.config.podcast_max_sentences_per_turn
        if len(sentences) > max_s:
            cleaned = " ".join(sentences[:max_s])
            
        return cleaned

