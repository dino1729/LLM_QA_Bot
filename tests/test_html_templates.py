"""
Tests for newsletter HTML rendering helpers.
"""
from helper_functions.html_templates import render_lesson_html


class TestRenderLessonMemoryPalace:
    """The Memory Palace section must only render real, meaningful insights."""

    def test_renders_real_memory_palace_insight(self):
        html = render_lesson_html({
            "key_insight": "Visionary leadership architects scalable systems.",
            "mp_insight": "As AI commoditizes answers, human value shifts to abstraction and reasoning.",
            "mp_topic": "Economics & Finance",
        })
        assert "MEMORY PALACE" in html
        assert "human value shifts to abstraction" in html

    def test_skips_placeholder_memory_palace_insight(self):
        """A placeholder mp_insight ('...') must not produce an empty section.

        Regression test: the 2026-06-10/06-11 newsletters rendered
        "🏛️ MEMORY PALACE — Observations & Ideas" with a body of just "...".
        """
        html = render_lesson_html({
            "key_insight": "Visionary leadership architects scalable systems.",
            "mp_insight": "...",
            "mp_topic": "Observations & Ideas",
        })
        assert "MEMORY PALACE" not in html

    def test_skips_whitespace_memory_palace_insight(self):
        html = render_lesson_html({
            "key_insight": "A real key insight that should still render.",
            "mp_insight": "   ",
            "mp_topic": "Observations & Ideas",
        })
        assert "MEMORY PALACE" not in html
        # The rest of the lesson must still render.
        assert "KEY INSIGHT" in html

    def test_no_memory_palace_key_renders_nothing(self):
        html = render_lesson_html({
            "key_insight": "Just a generated lesson, no Memory Palace blend.",
        })
        assert "MEMORY PALACE" not in html
