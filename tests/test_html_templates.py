"""
Tests for newsletter HTML rendering helpers.
"""
from helper_functions.html_templates import (
    render_lesson_html,
    render_newsletter_html_from_bundle,
)


def _bundle_with_news(newsletter):
    """Minimal bundle for exercising render_newsletter_html_from_bundle."""
    return {
        "meta": {"date_formatted": "June 11, 2026", "day_of_week": "Thursday"},
        "news": {"newsletter": newsletter},
    }


class TestRenderNewsletterHandlesMissingSections:
    """The news renderer must degrade gracefully when sections are absent.

    News generation sets sections to None when it is skipped or fails. The
    renderer previously crashed with AttributeError on None.get(...).
    """

    def test_none_sections_does_not_crash(self):
        bundle = _bundle_with_news(
            {"sections": None, "voicebot_script": None, "podcast_transcript": []}
        )
        html = render_newsletter_html_from_bundle(bundle)
        assert "Daily Briefing" in html
        assert "No updates available" in html

    def test_missing_sections_key_does_not_crash(self):
        bundle = _bundle_with_news({"voicebot_script": None})
        html = render_newsletter_html_from_bundle(bundle)
        assert "Daily Briefing" in html

    def test_none_newsletter_does_not_crash(self):
        bundle = _bundle_with_news(None)
        html = render_newsletter_html_from_bundle(bundle)
        assert "Daily Briefing" in html

    def test_populated_sections_still_render(self):
        bundle = _bundle_with_news({
            "sections": {
                "tech": [{"source": "TechCo", "headline": "AI breakthrough", "commentary": "Big."}],
                "financial": [],
                "india": [],
            }
        })
        html = render_newsletter_html_from_bundle(bundle)
        assert "AI breakthrough" in html


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
