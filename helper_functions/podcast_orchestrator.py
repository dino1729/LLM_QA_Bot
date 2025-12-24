import os
import logging
from typing import List, Any
import numpy as np
from pathlib import Path

from helper_functions.dialogue_engine import DialogueEngine, DialogueTurn
from helper_functions.tts_chatterbox import get_chatterbox_tts
from helper_functions.audio_processors import assemble_podcast_audio

logger = logging.getLogger(__name__)

class PodcastOrchestrator:
    """
    Orchestrates the generation of a two-persona podcast news briefing.
    """
    
    def __init__(self, config: Any, use_chatterbox: bool = True):
        self.config = config
        self.use_chatterbox = use_chatterbox
        self.engine = DialogueEngine(config, use_chatterbox=use_chatterbox)
        self.output_dir = Path("newsletter_research_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TTS settings from config - for expressive podcast delivery
        # Lower cfg_weight + higher exaggeration = more expressive, dramatic speech
        self.podcast_cfg_weight = getattr(config, 'podcast_tts_cfg_weight', None)
        self.podcast_exaggeration = getattr(config, 'podcast_tts_exaggeration', None)
        self.default_voice = getattr(config, 'chatterbox_tts_default_voice', None)
        
        # Validate required config
        if self.podcast_cfg_weight is None:
            raise ValueError("Missing required config: 'podcast_tts_cfg_weight' in config.yml")
        if self.podcast_exaggeration is None:
            raise ValueError("Missing required config: 'podcast_tts_exaggeration' in config.yml")
        if self.default_voice is None:
            raise ValueError("Missing required config: 'chatterbox_tts_default_voice' in config.yml (used as fallback when character voice is not found)")

    def generate_podcast(self, topic: str, context: str, output_filename: str = "news_podcast.wav", pregenerated_turns: List[DialogueTurn] = None) -> tuple[str, List[DialogueTurn]]:
        """
        Generates the full podcast audio and returns the path to the file and the turns.
        """
        # 1. Generate Dialogue or use pre-generated
        if pregenerated_turns:
            print("🎙️ Using pre-generated podcast dialogue turns from cache...")
            turns = pregenerated_turns
        else:
            turns = self.engine.run_conversation(topic, context)
            
        if not turns:
            logger.error("No dialogue turns available")
            return "", []

        # 2. TTS Synthesis
        print(f"🔊 Synthesizing {len(turns)} turns using Chatterbox...")
        turns_audio = []
        tts = get_chatterbox_tts(model_type=self.config.chatterbox_tts_model_type)
        
        for i, turn in enumerate(turns):
            print(f"  Synthesizing turn {i+1}/{len(turns)} ({turn.character_id})...")
            
            # Use character reference audio from voices/ folder
            voice_path = Path(f"voices/{turn.character_id}.wav")
            if not voice_path.exists():
                logger.warning(f"Voice file not found for {turn.character_id}: {voice_path}. Using fallback.")
                voice_path = Path(f"voices/{self.default_voice}.wav")

            # Use expressive TTS settings for dramatic podcast delivery
            # Lower cfg_weight (~0.3) + higher exaggeration (~0.7) = more expressive speech
            # Higher exaggeration speeds up speech; lower cfg_weight compensates with slower pacing
            audio_data = tts.synthesize(
                turn.text,
                audio_prompt_path=str(voice_path),
                cfg_weight=self.podcast_cfg_weight,
                exaggeration=self.podcast_exaggeration,
            )
            
            if audio_data is not None and len(audio_data) > 0:
                turns_audio.append(audio_data)
            else:
                logger.error(f"TTS failed for turn {i+1}")

        if not turns_audio:
            logger.error("No audio generated for any turns")
            return "", turns

        # 3. Assemble Audio
        print("🎵 Assembling podcast audio pipeline...")
        final_audio = assemble_podcast_audio(turns_audio, tts.sample_rate, self.config)
        
        # 4. Save Final Output
        output_path = self.output_dir / output_filename
        success = tts.save_audio(final_audio, str(output_path))
        
        if success:
            duration = len(final_audio) / tts.sample_rate
            print(f"✅ Podcast generated: {duration:.1f}s saved to {output_path}")
            return str(output_path), turns
        else:
            logger.error("Failed to save final podcast audio")
            return "", turns

