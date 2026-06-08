// Reddit Search Engine — main app

const state = {
  query: '',
  sort: 'relevance',
  time: 'all',
  subreddit: '',
  posts: [],
  after: null,
  loading: false,
};

const el = {};

// ── Utilities ──────────────────────────────────────────────────────────────────

function fmt(n) {
  n = Number(n) || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(n);
}

function ago(ts) {
  const d = Math.floor(Date.now() / 1000 - ts);
  if (d < 60) return 'just now';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  if (d < 604800) return Math.floor(d / 86400) + 'd ago';
  if (d < 2592000) return Math.floor(d / 604800) + 'w ago';
  return Math.floor(d / 2592000) + 'mo ago';
}

function subHue(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function subColors(name) {
  const hue = subHue(name);
  return {
    bg:       'hsl(' + hue + ',50%,92%)',
    text:     'hsl(' + hue + ',55%,30%)',
    icon:     'hsl(' + hue + ',55%,44%)',
    darkBg:   'hsl(' + hue + ',35%,16%)',
    darkText: 'hsl(' + hue + ',55%,72%)',
    darkIcon: 'hsl(' + hue + ',50%,52%)',
  };
}

function initial(name) {
  return (name || '?')[0].toUpperCase();
}

function trunc(text, max) {
  max = max || 220;
  if (!text) return '';
  const s = text.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
  return s.length > max ? s.slice(0, max).trimEnd() + '…' : s;
}

function demd(text) {
  return (text || '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/\[(.+?)\]\(.+?\)/g, '$1')
    .replace(/^#+\s*/gm, '')
    .replace(/^[-*>]\s*/gm, '')
    .replace(/`(.+?)`/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/\n+/g, ' ')
    .trim();
}

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function validThumb(url) {
  return typeof url === 'string' && url.startsWith('https://') && url.length > 12;
}

// ══════════════════════════════════════════════════════════════════════════════
//  KNOWLEDGE BAR ENGINE
//  Three-stage pipeline:
//    1. detectIntent(query)  → classify query into 7 intent types
//    2. pickBestPost(posts)  → score all top-5 posts, pick the best candidate
//    3. pickBestComment()    → multi-factor comment scoring with intent alignment
// ══════════════════════════════════════════════════════════════════════════════

// Intent config: label + accent color for panel UI
const INTENT_CONFIG = {
  howto:      { label: 'How-To Guide',         accent: '#0EA5E9' },
  comparison: { label: 'Community Comparison', accent: '#7C3AED' },
  recommend:  { label: 'Top Picks',            accent: '#D97706' },
  definition: { label: 'What Is…',             accent: '#059669' },
  problem:    { label: 'Community Fix',        accent: '#DC2626' },
  question:   { label: 'Featured Answer',      accent: '#FF4500' },
  general:    { label: 'Top Discussion',       accent: '#FF4500' },
};

// ── Stage 1: Intent Detection ─────────────────────────────────────────────────
//
// Evaluated in priority order — first match wins.
// Patterns are deliberately broad to catch natural-language phrasings.

function detectIntent(query) {
  const q = query.trim();

  if (/\bhow\s+to\b|step[- ]by[- ]step|guide (to|for)|tutorial|learn (to|how)|ways?\s+to\b/i.test(q))
    return 'howto';

  if (/\bvs\.?\b|versus|\bor\b.{1,40}(better|worse)|compar(e|ed|ing) to|difference between|which is better/i.test(q))
    return 'comparison';

  if (/\b(best|top \d*|recommend(ed)?|worth (it|buying|getting)|should i (get|buy|use|try)|which .{1,20}(to (get|use|buy)))\b/i.test(q))
    return 'recommend';

  if (/\bwhat (is|are|was|were)\b|define\b|meaning of\b|what does .{1,30} mean|explain\b/i.test(q))
    return 'definition';

  if (/\b(fix|solve|not working|broken|error|issue|problem|crash|bug|help with|can'?t|cannot|won'?t|doesn'?t work)\b/i.test(q))
    return 'problem';

  if (/^(how|what|why|when|where|which|who|can|should|is|are|does|do|will|would|was|were)\b/i.test(q))
    return 'question';

  return 'general';
}

// ── Stage 2: Post Scoring ─────────────────────────────────────────────────────
//
// Score each post candidate for fitness as a knowledge panel source.
// Uses log-scaling on the Reddit score to prevent viral outliers dominating.

function scorePost(post, intent) {
  // Log-scaled base: dampens the extreme variance in Reddit scores
  let s = Math.log10(Math.max(post.score, 1) + 1) * 10;

  // Upvote ratio is a clean community quality signal (0→0.5x, 1→1.5x)
  s *= (0.5 + (post.upvote_ratio || 0.5));

  // Comment engagement: more discussion = richer comment pool
  const c = post.num_comments || 0;
  if (c >= 5)   s *= 1.12;
  if (c >= 25)  s *= 1.10;
  if (c >= 100) s *= 1.08;

  // Awards are a rare, high-signal community endorsement
  const awards = post.total_awards_received || 0;
  if (awards >= 1)  s *= 1.08;
  if (awards >= 3)  s *= 1.07;
  if (awards >= 10) s *= 1.05;

  // Self/text posts have actual content + discussion, better for panels
  if (post.is_self && post.selftext && post.selftext.length > 150) s *= 1.30;
  else if (post.is_self) s *= 1.08;

  // Intent-specific post preferences
  if (intent === 'howto' && post.is_self)       s *= 1.35; // tutorials live in text posts
  if (intent === 'recommend' && c >= 50)        s *= 1.25; // recommendations need crowd input
  if (intent === 'problem' && post.is_self)     s *= 1.20; // problem posts need context
  if (intent === 'definition' && post.is_self)  s *= 1.15;

  // Age penalty/bonus: penalize too-fresh (unvalidated) and very old (outdated)
  const ageHours = (Date.now() / 1000 - post.created_utc) / 3600;
  if (ageHours < 1)           s *= 0.35;
  else if (ageHours < 6)      s *= 0.60;
  else if (ageHours < 24)     s *= 0.82;
  else if (ageHours > 26280)  s *= 0.88; // > 3 years: possibly stale

  return s;
}

function pickBestPost(posts, intent) {
  // Only consider top-5 results — beyond that, results drift off-topic
  const candidates = posts.slice(0, 5);
  let best = null;
  let bestScore = -Infinity;
  for (const p of candidates) {
    const s = scorePost(p, intent);
    if (s > bestScore) { bestScore = s; best = p; }
  }
  // Require minimum Reddit validation before showing a panel
  return (best && best.score >= 15) ? best : null;
}

// ── Stage 3: Comment Scoring ──────────────────────────────────────────────────
//
// Multi-factor quality scoring with intent alignment.
// Modelled as: base_score × length_multiplier × structure_bonus × intent_bonus × penalties

function scoreComment(comment, intent) {
  const raw  = comment.body || '';
  const text = demd(raw);

  // Hard disqualifications
  if (!raw || raw === '[deleted]' || raw === '[removed]') return -1;
  if (comment.author === '[deleted]' || comment.author === 'AutoModerator') return -1;
  if (comment.stickied) return -1;

  // Log-scaled base from Reddit score
  let s = Math.log10(Math.max(comment.score, 1) + 1) * 10;

  // ── Length scoring (ideal 120–700 chars) ────────────────────────────────────
  const len = text.length;
  if      (len < 40)             s *= 0.04;
  else if (len < 80)             s *= 0.35;
  else if (len < 120)            s *= 0.68;
  else if (len >= 120 && len <= 700) s *= 1.45;
  else if (len <= 1400)          s *= 1.18;
  else if (len <= 2500)          s *= 0.88;
  else                           s *= 0.55;  // TL;DR territory

  // ── Structure bonuses ───────────────────────────────────────────────────────
  const hasNumbered  = /^\d+\s*[.)]/m.test(raw);
  const hasBullets   = /^[*-]\s+\S/m.test(raw);
  const hasParagraph = (raw.match(/\n\n/g) || []).length >= 1;
  const hasHeaders   = /^#+\s/m.test(raw);

  if (hasNumbered)  s *= 1.38; // numbered lists signal structured, authoritative answers
  if (hasBullets)   s *= 1.20;
  if (hasParagraph) s *= 1.15;
  if (hasHeaders)   s *= 1.12;

  // ── Awards: exceptional community endorsement ────────────────────────────────
  const awards = comment.total_awards_received || 0;
  if (awards >= 1) s *= (1 + Math.min(awards, 5) * 0.15);

  // ── Positive semantic signals ────────────────────────────────────────────────
  const lc = text.toLowerCase().slice(0, 300); // check only opening for efficiency

  // Authoritative openers suggest the commenter is confident and direct
  if (/^(the (key|trick|answer|secret|solution|reason|issue)|here'?s? (how|why)|you (need|want|should)|in my experience|this (is|works?)|the (problem|fix|way) is)/i.test(text))
    s *= 1.22;

  // Causal/explanatory language signals depth
  if (/\b(because|therefore|since|as a result|which means|that'?s? why|the reason)\b/i.test(text))
    s *= 1.12;

  // ── Negative signals ─────────────────────────────────────────────────────────

  // Pure quote-reply with little original content
  if (raw.trim().startsWith('>') && len < 200) s *= 0.40;

  // Reaction / social fluff (not informational)
  if (/^(lol|haha|wtf|same|this!?|rip |omg|wow|nice!?|yep|nope|true|exactly|agreed|indeed|fair|^ {0,2}f$)/i.test(text.slice(0, 20)))
    s *= 0.08;

  // Deflects rather than answers
  if (/^(did you|have you tried|what (did|do) you|why did you|are you sure)/i.test(text))
    s *= 0.35;

  // Link-dump without explanation
  if ((raw.match(/https?:\/\//g) || []).length >= 3 && len < 200) s *= 0.42;

  // Wishy-washy non-answers (only penalised if brief)
  if (/^(it depends|kind of|sort of|maybe|idk|not sure|hard to say)\b/i.test(text) && len < 120)
    s *= 0.45;

  // ── Intent alignment ─────────────────────────────────────────────────────────

  if (intent === 'howto') {
    if (hasNumbered) s *= 1.45; // step-by-step lists are gold for how-to
    if (/\b(step|first[,.\s]|then[,.\s]|next[,.\s]|finally[,.\s]|lastly[,.\s])\b/i.test(lc)) s *= 1.28;
    if (/\b(you'?ll? need|make sure|remember (to|that)|important(ly)?|note:)\b/i.test(lc)) s *= 1.12;
  }

  if (intent === 'recommend') {
    if (/\bi (use|used|recommend|suggest|prefer|switched to|'?ve been using|'?ve tried)\b/i.test(lc)) s *= 1.38;
    if (/\b(the best|my (top|go-to|favorite)|highly recommend|can'?t go wrong|solid choice)\b/i.test(lc)) s *= 1.22;
  }

  if (intent === 'definition') {
    // Good definitions start by establishing what the thing IS
    if (/^.{0,60}(is (a|an|the)|refers? to|means?|defined? as|basically (a|an)|essentially)/i.test(text)) s *= 1.45;
  }

  if (intent === 'problem') {
    if (/\b(fix(ed)?|solution|solved?|try (this|the)|worked for me|the (issue|problem|bug) (was|is))\b/i.test(lc)) s *= 1.35;
    if (/\b(edit|update):?.{0,20}(fixed|solved|worked)/i.test(lc)) s *= 1.25; // confirmed solution
  }

  if (intent === 'comparison') {
    if (/\b(pros?|cons?|advantage|disadvantage|trade.?off|better (for|at|when)|worse (for|at)|prefer.+over)\b/i.test(lc)) s *= 1.30;
    if (hasBullets || hasNumbered) s *= 1.20; // structured comparisons are clearer
  }

  if (intent === 'question' || intent === 'general') {
    // Explanatory depth is rewarded for general Q&A
    if (len >= 200 && hasParagraph) s *= 1.18;
  }

  return s;
}

function pickBestComment(comments, intent) {
  let best = null;
  let bestScore = -Infinity;
  for (const c of comments) {
    const s = scoreComment(c, intent);
    if (s > bestScore) { bestScore = s; best = c; }
  }
  // Must meet minimum quality bar to display
  const minLen = demd((best || {}).body || '').length;
  return (best && best.score >= 2 && minLen >= 60) ? best : null;
}

// ── Knowledge Bar: Orchestrator ───────────────────────────────────────────────

async function buildKnowledge(posts, intent) {
  const post = pickBestPost(posts, intent);
  if (!post) return;

  // Show skeleton immediately so the panel space is claimed
  const cfg = INTENT_CONFIG[intent] || INTENT_CONFIG.general;
  el.knowledgePanel.style.setProperty('--k-accent', cfg.accent);
  el.knowledgePanel.innerHTML =
    '<div class="k-skeleton">' +
      '<div class="skel-line" style="width:38%"></div>' +
      '<div class="skel-line" style="width:65%"></div>' +
      '<div class="skel-line" style="width:95%"></div>' +
      '<div class="skel-line" style="width:88%"></div>' +
      '<div class="skel-line" style="width:55%"></div>' +
    '</div>';
  el.knowledgePanel.classList.remove('hidden');

  try {
    const [commentRes, subRes] = await Promise.allSettled([
      RedditAPI.getComments(post.subreddit, post.id),
      RedditAPI.getSubreddit(post.subreddit),
    ]);

    const comments = commentRes.status === 'fulfilled' ? commentRes.value.comments : [];
    const subData  = subRes.status === 'fulfilled'     ? subRes.value             : null;

    const html = renderKnowledge(post, comments, subData, intent);
    if (html) {
      el.knowledgePanel.innerHTML = html;
    } else {
      el.knowledgePanel.classList.add('hidden');
    }
  } catch (_) {
    el.knowledgePanel.classList.add('hidden');
  }
}

// ── Knowledge Bar: Renderer ───────────────────────────────────────────────────

function renderKnowledge(post, comments, sub, intent) {
  const bestComment = pickBestComment(comments, intent);
  if (!bestComment && !post.selftext) return null;

  const cfg      = INTENT_CONFIG[intent] || INTENT_CONFIG.general;
  const subName  = esc(post.subreddit);
  const c        = subColors(post.subreddit);
  const postUrl  = 'https://www.reddit.com' + post.permalink;
  const subUrl   = 'https://www.reddit.com/r/' + subName + '/';
  const members  = sub && sub.subscribers ? fmt(sub.subscribers) + ' members' : '';
  const subDesc  = sub && sub.public_description
    ? '<p class="k-sub-desc">' + esc(trunc(demd(sub.public_description), 120)) + '</p>'
    : '';

  const subStyle =
    '--sub-bg:'     + c.bg       + ';' +
    '--sub-text:'   + c.text     + ';' +
    '--sub-icon:'   + c.icon     + ';' +
    '--sub-bg-d:'   + c.darkBg   + ';' +
    '--sub-text-d:' + c.darkText + ';' +
    '--sub-icon-d:' + c.darkIcon;

  let answer = '';
  if (bestComment) {
    const body = esc(trunc(demd(bestComment.body), 520));
    const commentScore = fmt(bestComment.score);
    const commentAge   = ago(bestComment.created_utc);
    const hasAwards    = (bestComment.total_awards_received || 0) > 0;
    const awardsHtml   = hasAwards
      ? ' <span class="k-comment-awards">✨ ' + bestComment.total_awards_received + '</span>'
      : '';

    answer =
      '<div class="k-answer">' +
        '<div class="k-answer-meta">' +
          '<span class="k-answer-label">' + cfg.label + '</span>' +
          '<span class="k-answer-sep">·</span>' +
          '<span class="k-answer-by">' +
            '<a href="https://www.reddit.com/u/' + esc(bestComment.author) + '/" target="_blank" rel="noopener noreferrer">' +
              'u/' + esc(bestComment.author) +
            '</a>' +
          '</span>' +
          '<span class="k-score">▲ ' + commentScore + awardsHtml + '</span>' +
          '<span class="k-comment-age">' + commentAge + '</span>' +
        '</div>' +
        '<p class="k-answer-body">' + body + '</p>' +
        '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-read-more">Read full discussion →</a>' +
      '</div>';
  } else if (post.selftext) {
    const body = esc(trunc(demd(post.selftext), 460));
    answer =
      '<div class="k-answer">' +
        '<div class="k-answer-meta">' +
          '<span class="k-answer-label">' + cfg.label + '</span>' +
        '</div>' +
        '<p class="k-answer-body">' + body + '</p>' +
        '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-read-more">Read full post →</a>' +
      '</div>';
  }

  const postAge   = ago(post.created_utc);
  const postScore = fmt(post.score);

  return (
    '<div class="k-header">' +
      '<a href="' + subUrl + '" target="_blank" rel="noopener noreferrer" class="k-sub" style="' + subStyle + '">' +
        '<span class="k-sub-icon">' + initial(post.subreddit) + '</span>' +
        '<div class="k-sub-info">' +
          '<span class="k-sub-name">r/' + subName + '</span>' +
          (members ? '<span class="k-sub-members">' + members + '</span>' : '') +
        '</div>' +
      '</a>' +
      '<span class="k-intent-tag" style="background:' + esc(cfg.accent) + '">' + esc(cfg.label) + '</span>' +
    '</div>' +
    subDesc +
    '<div class="k-post-ref">' +
      '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer">' + esc(post.title) + '</a>' +
      '<span class="k-post-stats">▲ ' + postScore + ' · ' + postAge + '</span>' +
    '</div>' +
    answer +
    '<div class="k-actions">' +
      '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-btn primary">View Discussion</a>' +
      '<a href="' + subUrl + '" target="_blank" rel="noopener noreferrer" class="k-btn">r/' + subName + '</a>' +
    '</div>'
  );
}

// ══════════════════════════════════════════════════════════════════════════════
//  RESULT CARD RENDERER
// ══════════════════════════════════════════════════════════════════════════════

function renderCard(p) {
  const sub   = esc(p.subreddit);
  const c     = subColors(p.subreddit);
  const url   = 'https://www.reddit.com' + p.permalink;
  const thumb = validThumb(p.thumbnail) ? p.thumbnail : null;

  let snippet = '';
  if (p.selftext) {
    snippet = '<p class="card-snippet">' + esc(trunc(demd(p.selftext))) + '</p>';
  } else if (p.url && !p.url.includes('reddit.com')) {
    try {
      const host = new URL(p.url).hostname.replace('www.', '');
      snippet = '<p class="card-snippet link-url">&#128279; ' + esc(host) + '</p>';
    } catch (_) { /* invalid url */ }
  }

  const awards = p.total_awards_received > 0
    ? '<span class="award-badge">&#10024; ' + p.total_awards_received + '</span>'
    : '';

  const flair = p.link_flair_text
    ? '<span class="flair-tag">' + esc(p.link_flair_text) + '</span>'
    : '';

  const style =
    '--sub-bg:'     + c.bg       + ';' +
    '--sub-text:'   + c.text     + ';' +
    '--sub-icon:'   + c.icon     + ';' +
    '--sub-bg-d:'   + c.darkBg   + ';' +
    '--sub-text-d:' + c.darkText + ';' +
    '--sub-icon-d:' + c.darkIcon;

  return (
    '<article class="result-card">' +
      (thumb
        ? '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer" class="card-thumb-wrap">' +
            '<img class="card-thumb" src="' + esc(thumb) + '" alt="" loading="lazy" ' +
            'onerror="this.closest(\'.card-thumb-wrap\').remove()">' +
          '</a>'
        : '') +
      '<div class="card-body">' +
        '<div class="card-meta">' +
          '<a href="https://www.reddit.com/r/' + sub + '/" target="_blank" rel="noopener noreferrer" ' +
             'class="sub-pill" style="' + style + '">' +
            '<span class="sub-dot">' + initial(p.subreddit) + '</span>' +
            'r/' + sub +
          '</a>' +
          '<span class="meta-dot">&#183;</span>' +
          '<span class="meta-auth">u/' + esc(p.author) + '</span>' +
          '<span class="meta-dot">&#183;</span>' +
          '<time class="meta-time">' + ago(p.created_utc) + '</time>' +
          flair +
        '</div>' +
        '<h2 class="card-title">' +
          '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + esc(p.title) + '</a>' +
        '</h2>' +
        snippet +
        '<div class="card-stats">' +
          '<span class="stat score">&#9650; ' + fmt(p.score) + '</span>' +
          '<span class="stat comments">&#128172; ' + fmt(p.num_comments) + '</span>' +
          awards +
          '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer" class="view-btn">View on Reddit &#8594;</a>' +
        '</div>' +
      '</div>' +
    '</article>'
  );
}

// ══════════════════════════════════════════════════════════════════════════════
//  VIEWS / URL STATE / LOADING
// ══════════════════════════════════════════════════════════════════════════════

function showHome() {
  el.homeView.classList.remove('hidden');
  el.resultsView.classList.add('hidden');
  el.homeInput.focus();
  document.title = 'RedditSearch — Find anything on Reddit';
  history.pushState({}, '', location.pathname);
}

function showResults() {
  el.homeView.classList.add('hidden');
  el.resultsView.classList.remove('hidden');
}

function pushURL() {
  const p = new URLSearchParams();
  if (state.query) p.set('q', state.query);
  if (state.sort !== 'relevance') p.set('sort', state.sort);
  if (state.time !== 'all') p.set('t', state.time);
  if (state.subreddit) p.set('sub', state.subreddit);
  history.pushState({}, '', '?' + p);
}

function readURL() {
  const p = new URLSearchParams(location.search);
  state.query     = p.get('q')    || '';
  state.sort      = p.get('sort') || 'relevance';
  state.time      = p.get('t')    || 'all';
  state.subreddit = p.get('sub')  || '';
  return !!state.query;
}

function setPill(filter, value) {
  document.querySelectorAll('[data-filter="' + filter + '"]').forEach(function (b) {
    b.classList.toggle('active', b.dataset.value === value);
  });
}

function setLoading(on) {
  state.loading = on;
  el.loadingState.classList.toggle('hidden', !on);
  if (on) {
    el.errorState.classList.add('hidden');
    el.noResults.classList.add('hidden');
  }
}

function showErr(msg) {
  el.errorState.innerHTML =
    '<div class="state-icon">&#9888;&#65039;</div>' +
    '<h3>Something went wrong</h3>' +
    '<p>' + esc(msg) + '</p>' +
    '<button onclick="search(true)" class="retry-btn">Try again</button>';
  el.errorState.classList.remove('hidden');
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN SEARCH
// ══════════════════════════════════════════════════════════════════════════════

async function search(reset) {
  if (reset === undefined) reset = true;
  if (!state.query.trim()) return;

  showResults();
  setLoading(true);

  el.resultsInput.value   = state.query;
  el.subredditInput.value = state.subreddit;
  document.title = state.query + ' — RedditSearch';

  setPill('sort', state.sort);
  setPill('time', state.time);

  if (reset) {
    state.posts = [];
    state.after = null;
    el.resultsList.innerHTML    = '';
    el.statsBar.textContent     = '';
    el.knowledgePanel.classList.add('hidden');
    el.knowledgePanel.innerHTML = '';
    el.loadMoreWrap.classList.add('hidden');
  }

  pushURL();

  try {
    const res = await RedditAPI.search({
      query:     state.query,
      sort:      state.sort,
      time:      state.time,
      limit:     25,
      after:     reset ? null : state.after,
      subreddit: state.subreddit || null,
    });

    state.after = res.after;
    state.posts = reset ? res.posts : state.posts.concat(res.posts);

    setLoading(false);

    if (state.posts.length === 0 && reset) {
      el.noResults.classList.remove('hidden');
      return;
    }

    el.statsBar.textContent = fmt(state.posts.length) + (res.after ? '+' : '') + ' results';

    res.posts.forEach(function (p) {
      el.resultsList.insertAdjacentHTML('beforeend', renderCard(p));
    });

    el.loadMoreWrap.classList.toggle('hidden', !res.after);

    // Launch knowledge panel on first page only, using detected intent
    if (reset && res.posts.length > 0) {
      const intent = detectIntent(state.query);
      buildKnowledge(res.posts, intent);
    }
  } catch (err) {
    setLoading(false);
    const msg = String(err.message || '');
    if (err.status === 429) {
      showErr('Reddit is rate-limiting requests. Wait a minute and try again.');
    } else if (err.status === 403) {
      showErr('Reddit blocked this request (403 Forbidden). Try opening reddit.com in a new tab to clear any challenge, then search again.');
    } else if (err.status === 0 || msg.includes('connect') || msg.includes('internet')) {
      showErr(msg || 'Could not reach Reddit. Check your internet connection and try again.');
    } else {
      showErr(msg || 'Unexpected error — please try again.');
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', function () {
  el.homeView       = document.getElementById('home-view');
  el.resultsView    = document.getElementById('results-view');
  el.homeForm       = document.getElementById('home-form');
  el.homeInput      = document.getElementById('home-input');
  el.resultsForm    = document.getElementById('results-form');
  el.resultsInput   = document.getElementById('results-input');
  el.logoBtn        = document.getElementById('logo-btn');
  el.knowledgePanel = document.getElementById('knowledge-panel');
  el.resultsList    = document.getElementById('results-list');
  el.loadingState   = document.getElementById('loading-state');
  el.errorState     = document.getElementById('error-state');
  el.noResults      = document.getElementById('no-results');
  el.statsBar       = document.getElementById('stats-bar');
  el.loadMoreWrap   = document.getElementById('load-more-wrap');
  el.loadMoreBtn    = document.getElementById('load-more-btn');
  el.subredditInput = document.getElementById('subreddit-input');

  function submitFrom(input) {
    return function (e) {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;
      state.query = q;
      state.sort  = 'relevance';
      state.time  = 'all';
      search(true);
    };
  }

  el.homeForm.addEventListener('submit', submitFrom(el.homeInput));
  el.resultsForm.addEventListener('submit', submitFrom(el.resultsInput));
  el.logoBtn.addEventListener('click', showHome);

  document.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      state.query     = chip.dataset.q;
      state.sort      = 'relevance';
      state.time      = 'all';
      state.subreddit = '';
      search(true);
    });
  });

  document.querySelector('.filter-bar').addEventListener('click', function (e) {
    const btn = e.target.closest('[data-filter]');
    if (!btn) return;
    if (btn.dataset.filter === 'sort') state.sort = btn.dataset.value;
    else if (btn.dataset.filter === 'time') state.time = btn.dataset.value;
    if (state.query) search(true);
  });

  var subTimer;
  el.subredditInput.addEventListener('input', function (e) {
    clearTimeout(subTimer);
    subTimer = setTimeout(function () {
      state.subreddit = e.target.value.replace(/^r\//, '').trim();
      if (state.query) search(true);
    }, 700);
  });

  el.subredditInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      clearTimeout(subTimer);
      state.subreddit = el.subredditInput.value.replace(/^r\//, '').trim();
      if (state.query) search(true);
    }
  });

  el.loadMoreBtn.addEventListener('click', function () { search(false); });

  window.addEventListener('popstate', function () {
    if (readURL()) search(true);
    else showHome();
  });

  if (readURL()) search(true);
  else el.homeInput.focus();
});
