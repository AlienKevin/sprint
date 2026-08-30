"""The public setup summary stays concise and distinguishes local/official tests."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_setup_precedes_observations_and_describes_three_safeguards() -> None:
    page = (ROOT / "web/index.html").read_text()
    setup = re.search(r'<section[^>]+id="setup">([\s\S]*?)</section>', page).group(1)
    assert page.index('id="setup"') < page.index('id="cost-breakdown"') < page.index('id="scoring-guide"') < page.index('id="observations"')
    assert '<h2>Setup</h2>' in setup
    assert '<h3>Integrity</h3>' in setup
    assert setup.count('<li>') == 3
    for text in ['PyTorch', 'Isaac Lab', 'G1 robot assets', 'course and scoring code',
                 'only OpenRouter', 'model and provider', 'receipts, not performance results',
                 'on their own budget', 'test and debug policies', 'logging performance and costs']:
        assert text in setup
    assert 'all internet access' not in setup
    assert 'identical results' not in setup
    assert '—' not in setup


def test_setup_blind_feedback_matches_current_benchmark_contract() -> None:
    task = (ROOT / "events/g1-100-metres/task.toml").read_text()
    assert 'return_results_to_agent = false' in task
    assert 'return_acknowledgments_to_agent = true' in task
    local = (ROOT / "event_runtime/agent/test_policy.py").read_text()
    assert 'exec bash /opt/event-verifier/test.sh' in local
    assert '"--job-kind",\n        "verify"' in local


def test_harnesses_document_actual_versions_and_settings_before_safeguards() -> None:
    page = (ROOT / "web/index.html").read_text()
    setup = re.search(r'<section[^>]+id="setup">([\s\S]*?)</section>', page).group(1)
    harnesses = re.search(r'<div class="harnesses-copy"[^>]*>([\s\S]*?)</div>', setup).group(1)
    assert setup.index('id="harnesses-heading"') < setup.index('<h3>Integrity</h3>')
    assert harnesses.count('<p>') == 4  # Introduction plus one paragraph per model.
    assert '<ol>' not in harnesses and '<li>' not in harnesses
    assert 'provider’s own benchmark setup' in harnesses
    assert 'shared $10 robot-training task' in harnesses
    paragraphs = re.findall(r'<p>([\s\S]*?)</p>', harnesses)
    deepseek, luna, glm = paragraphs[1:]
    assert re.sub(r'<[^>]+>', '', deepseek) == (
        'DeepSeek-V4-Flash runs in DeepSeek Harness 0.1.1-rc.2 in Minimal Mode '
        'following DeepSeek’s official benchmark setup.'
    )
    assert re.sub(r'<[^>]+>', '', luna) == (
        'GPT-5.6 Luna runs in Codex 0.149.1 with the pinned official Luna catalog.'
    )
    assert re.sub(r'<[^>]+>', '', glm) == (
        'GLM-5.3-Flash runs in Claude Code 2.1.248. Z.ai also uses Claude Code for '
        'Terminal-Bench 2.1 and Agents’ Last Exam.'
    )
    assert len(re.sub(r'<[^>]+>', ' ', harnesses).split()) <= 250


def test_harness_sources_use_shared_editorial_links_and_white_model_labels() -> None:
    page = (ROOT / "web/index.html").read_text()
    harnesses = re.search(r'<div class="harnesses-copy"[^>]*>([\s\S]*?)</div>', page).group(1)
    links = re.findall(r'<a class="text-link" href="([^"]+)"', harnesses)
    assert links == [
        'https://api-docs.deepseek.com/news/news260821/',
        'https://github.com/openai/codex/blob/rust-v0.149.1/codex-rs/models-manager/models.json',
        'https://z.ai/blog/glm-5.3-flash',
    ]
    assert re.findall(r'<strong>([^<]+)</strong>', harnesses) == [
        'DeepSeek-V4-Flash', 'GPT-5.6 Luna', 'GLM-5.3-Flash',
    ]
    css = (ROOT / 'web/styles.css').read_text()
    rule = re.search(r'\.harnesses-copy strong\s*\{([^}]+)\}', css).group(1)
    assert 'color: var(--text)' in rule
