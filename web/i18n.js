/* Explicit UI localization only. Model messages, source code and recordings are never translated. */
(() => {
  'use strict';
  const storageKey = 'agents100m.language';
  const catalogs = {en: Object.create(null), 'zh-CN': Object.create(null)};
  const normalize = value => value === 'zh-CN' || value === 'zh' ? 'zh-CN' : 'en';
  const known = value => ['en', 'zh', 'zh-CN'].includes(value);
  let language = 'en';
  try {
    const query = new URL(location.href).searchParams.get('lang');
    const parentLanguage = window.parent !== window ? window.parent.SiteI18n?.language : null;
    language = normalize(parentLanguage || (known(query) ? query : localStorage.getItem(storageKey)));
  } catch { /* Storage or cross-origin parents may be unavailable. English remains usable. */ }
  document.documentElement.lang = language;
  function register(locale, values) {
    if (!known(locale) || !values || typeof values !== 'object') return;
    Object.assign(catalogs[normalize(locale)], values);
  }
  function t(key, params = {}, fallback) {
    const value = catalogs[language][key] ?? catalogs.en[key] ?? fallback ?? key;
    return String(value).replace(/\{([\w.]+)\}/g, (match, name) => Object.hasOwn(params, name) ? String(params[name]) : match);
  }
  function apply(root = document) {
    const nodes = [...(root.matches?.('[data-i18n],[data-i18n-html],[data-i18n-attr]') ? [root] : []), ...root.querySelectorAll('[data-i18n],[data-i18n-html],[data-i18n-attr]')];
    for (const node of nodes) {
      if (node.dataset.i18n) node.textContent = t(node.dataset.i18n);
      // Only hand-authored catalog markup may use data-i18n-html. Never bind trace content.
      if (node.dataset.i18nHtml) node.innerHTML = t(node.dataset.i18nHtml);
      for (const pair of (node.dataset.i18nAttr || '').split(';')) {
        const split = pair.indexOf(':');
        if (split > 0) node.setAttribute(pair.slice(0, split).trim(), t(pair.slice(split + 1).trim()));
      }
    }
    for (const host of document.querySelectorAll('[data-language-switcher]')) mountSwitcher(host);
  }
  function send(target) { try { target.postMessage({type: 'site:language', language}, location.origin); } catch {} }
  function propagate(exclude) {
    for (const frame of document.querySelectorAll('iframe')) if (frame.contentWindow !== exclude) send(frame.contentWindow);
    if (window.parent !== window && window.parent !== exclude) send(window.parent);
  }
  function setLanguage(value, options = {}) {
    if (!known(value)) return;
    const next = normalize(value), changed = next !== language;
    language = next;
    document.documentElement.lang = language;
    // Only a deliberate choice writes shared storage. Rewriting received values
    // can resurrect older queued choices during a quick EN → 中 → EN sequence.
    if (options.persist !== false) try { localStorage.setItem(storageKey, language); } catch {}
    // A deliberate toggle takes precedence over a language in a shared URL on the next reload.
    try { const url = new URL(location.href); if (url.searchParams.has('lang')) { url.searchParams.set('lang', language); history.replaceState(history.state, '', url); } } catch {}
    if (changed) {
      apply();
      window.dispatchEvent(new CustomEvent('site:languagechange', {detail: {language}}));
    }
    if (options.broadcast !== false && (changed || options.broadcast)) propagate(options.exclude);
  }
  function mountSwitcher(host) {
    if (!host) return;
    host.classList.add('language-switcher');
    host.setAttribute('role', 'group');
    host.setAttribute('aria-label', t('ui.language'));
    if (!host.dataset.languageMounted) {
      host.dataset.languageMounted = 'true';
      for (const [locale, label, name] of [['en', 'EN', 'English'], ['zh-CN', '中', '简体中文']]) {
        const button = document.createElement('button');
        button.type = 'button'; button.textContent = label; button.lang = locale;
        button.dataset.language = locale; button.setAttribute('aria-label', name);
        button.addEventListener('click', () => setLanguage(locale));
        host.append(button);
      }
    }
    for (const button of host.querySelectorAll('[data-language]')) button.setAttribute('aria-pressed', String(button.dataset.language === language));
  }
  register('en', {'ui.language': 'Language'});
  register('zh-CN', {'ui.language': '语言'});
  window.SiteI18n = {t, register, apply, setLanguage, mountSwitcher, get language() { return language; }};
  window.addEventListener('storage', event => { if (event.key === storageKey && known(event.newValue)) setLanguage(event.newValue, {persist:false, broadcast:false}); });
  window.addEventListener('message', event => {
    if (event.origin !== location.origin) return;
    const trusted = event.source === window.parent || [...document.querySelectorAll('iframe')].some(frame => frame.contentWindow === event.source);
    if (!trusted) return;
    if (event.data?.type === 'site:language:request') send(event.source);
    // Relay through nested frames even when storage already synchronized this
    // window, but never reflect the message back to its sender.
    else if (event.data?.type === 'site:language' && known(event.data.language)) setLanguage(event.data.language, {persist:false, broadcast:true, exclude:event.source});
  });
  document.addEventListener('load', event => { if (event.target?.tagName === 'IFRAME') send(event.target.contentWindow); }, true);
  // Readiness is not a user choice: a loading child must never overwrite a
  // newer parent choice with the language it captured before assets loaded.
  const ready = () => { apply(); propagate(window.parent); if (window.parent !== window) try { window.parent.postMessage({type:'site:language:request'}, location.origin); } catch {} };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready, {once:true}); else ready();
})();
