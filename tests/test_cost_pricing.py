"""List token prices stay distinct from trial spend and preserve small prices."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web/app.js").read_text()
PRICING_PATH = ROOT / "web/assets/token-pricing.json"


def test_verified_provider_list_prices_have_provenance_and_no_discounts() -> None:
    pricing = json.loads(PRICING_PATH.read_text())
    assert pricing["basis"] == "undiscounted_list_price"
    assert pricing["unit"] == "USD per 1M tokens"
    assert pricing["as_of"] == "2026-08-28"
    assert pricing["verified_at"] == "2026-08-30"
    assert list(pricing["models"]) == ["deepseek", "luna", "glm"]
    for key, expected in {"deepseek": (.014, .44, 1.32), "luna": (.02, .20, 1.20), "glm": (.03, .15, .50)}.items():
        model = pricing["models"][key]
        assert tuple(model[field] for field in ["cached_input", "input", "output"]) == expected
        assert model["provider_api_url"] == f'https://openrouter.ai/api/v1/models/{model["model"]}/endpoints'
        assert model["provider_tag"]
        assert model["note"]
    assert pricing["models"]["luna"]["long_context"]["cached_input"] == .04
    assert pricing["models"]["luna"]["long_context"]["output"] == 1.80
    assert "json('/assets/token-pricing.json')" in APP
    assert "/data/token-pricing.json" not in APP


def test_price_columns_are_shared_per_model_in_best_and_all_trial_views() -> None:
    functions = APP[APP.index("  const MODEL ="):APP.index("  let chartResizeFrame")]
    script = "const assert=require('node:assert/strict');\n"
    script += f"const pricing=require({json.dumps(str(PRICING_PATH))});\n"
    script += "const window={};\n" + next(line for line in APP.splitlines() if line.strip().startswith("const t=")) + "\n"
    script += """
let showAllTrials=false,showAllBudgetTrials=false;
const document={querySelectorAll:()=>[]};
const target={innerHTML:''};
const $=selector=>selector.includes('.experiment-table-toggle')?null:target;
const trajectoryHref=runId=>`/trajectory?run=${runId}`;
""" + functions + r"""
state.pricing=pricing;
state.runs=Object.entries(pricing.models).flatMap(([key,model])=>Array.from({length:5},(_,i)=>({
  run_id:`${key}-${i+1}`,model:model.model,
  timeline:{comparison_summary:{final_api_cost_usd:1+i},resource_usage_summary:{}}
})));
const original=JSON.stringify(state);
for(const expanded of [false,true,false]){
  showAllBudgetTrials=expanded;renderResources();
  const html=target.innerHTML;
  assert.deepEqual([...html.matchAll(/<tbody class="budget-group ([^"]+)"/g)].map(m=>m[1]),['deepseek','luna','glm']);
  assert.equal((html.match(/class="budget-row /g)||[]).length,expanded?15:3);
  assert.equal((html.match(/class="budget-price"/g)||[]).length,6);
  assert.equal((html.match(/class="budget-price-mobile"/g)||[]).length,3);
  assert.equal((html.match(/class="budget-price-mobile-cell"/g)||[]).length,6);
  for(const price of ['$0.014','$1.32','$0.02','$1.20','$0.03','$0.50'])assert.ok(html.includes(`<strong>${price}</strong>`));
  assert.match(html,new RegExp(`class="budget-price" rowspan="${expanded?5:1}"`));
  assert.doesNotMatch(html,/<caption/);
  assert.doesNotMatch(html,/Benchmark spend in USD|Token list prices in USD per 1M tokens, excluding discounts/);
  assert.ok(html.includes(`List prices as of ${pricing.as_of} without OpenRouter discounts:`));
  assert.match(html,/>Cached input<\/th>/);
  assert.match(html,/>Output<\/th>/);
  assert.doesNotMatch(html,/>Input<\/th>/);
  assert.match(html,/Luna above 272,000 input tokens: \$0.04 cached input \/ \$1.80 output/);
  for(const cell of html.matchAll(/<td class="budget-price"[^>]*>/g))assert.doesNotMatch(cell[0],/--heat|budget-heat/);
}
assert.equal(JSON.stringify(state),original);
// Missing, unverified or different-model pricing must not become guessed rates.
state.pricing=null;renderResources();assert.doesNotMatch(target.innerHTML,/<strong>\$0.014<\/strong>/);
state.pricing={...pricing,basis:'realized_cost_per_token'};renderResources();assert.doesNotMatch(target.innerHTML,/<strong>\$0.014<\/strong>/);
state.pricing=pricing;state.runs[0].model='deepseek/future-model';renderResources();assert.doesNotMatch(target.innerHTML,/<strong>\$0.014<\/strong>/);
assert.equal(fmtTokenPrice(.014),'$0.014');assert.equal(fmtTokenPrice(.5),'$0.50');
assert.equal(fmtTokenPrice(0),'$0.00');assert.equal(fmtTokenPrice(null),'—');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mobile_prices_are_hidden_while_desktop_keeps_rate_cells() -> None:
    css = (ROOT / "web/styles.css").read_text()
    mobile = css.split("@media (max-width: 720px)", 1)[1]
    assert ".budget-row .budget-price { display: none; }" in mobile
    assert ".budget-price-mobile { display: none; }" in mobile
    assert ".budget-price-mobile { display: grid;" not in mobile
    assert ".budget-price-mobile { display: none; }" in css.split("@media", 1)[0]
    assert ".budget-row .budget-price { display: none; }" not in css.split("@media", 1)[0]


def test_cost_explanation_names_the_harnesses_behind_the_cached_tokens() -> None:
    html = (ROOT / "web/index.html").read_text()
    insights = html.split('class="cost-insights"', 1)[1].split("</div>", 1)[0]
    paragraphs = [part.split("</p>", 1)[0] for part in insights.split("<p>")[1:]]
    assert len(paragraphs) == 2
    assert paragraphs[0] == (
        "As the cost of flash models continues to drop due to inference engine and model architecture innovations, "
        "model API cost is now even lower than the CPU cost of running the agent for DeepSeek and GLM, "
        "allowing more GPU experiments within the same budget."
    )
    explanation = paragraphs[1]
    assert explanation.startswith("Harness design matters just as much as token cost. ")
    assert "DeepSeek’s cached input tokens cost less than half GLM’s" in explanation
    assert "yet DeepSeek spent nearly 2× as much on tokens in their best trials." in explanation
    assert "DeepSeek used DeepSeek Harness, which reread 74 million cached tokens across its best trial as context grew." in explanation
    assert "GLM’s Claude Code compacted near 168K tokens and read only 18 million." in explanation
    assert "Clever compactions can outweigh cheaper tokens." in explanation
    assert len(explanation.split()) <= 70  # Shortened copy plus the requested opening and best-trial scope.
