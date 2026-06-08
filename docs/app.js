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

function isQ(q) {
  return /^(how|what|why|when|where|which|who|can|should|is|are|does|do|will|would|was|were)\b/i.test(q.trim());
}

function validThumb(url) {
  return typeof url === 'string' && url.startsWith('https://') && url.length > 12;
}

// ── Views ──────────────────────────────────────────────────────────────────────

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

// ── URL state ─────────────────────────────────────────────────────────────────

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
  state.query    = p.get('q')    || '';
  state.sort     = p.get('sort') || 'relevance';
  state.time     = p.get('t')    || 'all';
  state.subreddit = p.get('sub') || '';
  return !!state.query;
}

// ── Filter pills ───────────────────────────────────────────────────────────────

function setPill(filter, value) {
  document.querySelectorAll('[data-filter="' + filter + '"]').forEach(function (b) {
    b.classList.toggle('active', b.dataset.value === value);
  });
}

// ── Loading / error ────────────────────────────────────────────────────────────

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
    '<div class="state-icon">⚠️</div>' +
    '<h3>Something went wrong</h3>' +
    '<p>' + esc(msg) + '</p>' +
    '<button onclick="search(true)" class="retry-btn">Try again</button>';
  el.errorState.classList.remove('hidden');
}

// ── Result card ────────────────────────────────────────────────────────────────

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
      snippet = '<p class="card-snippet link-url">🔗 ' + esc(host) + '</p>';
    } catch (_) { /* invalid url */ }
  }

  const awards = p.total_awards_received > 0
    ? '<span class="award-badge">✨ ' + p.total_awards_received + '</span>'
    : '';

  const flair = p.link_flair_text
    ? '<span class="flair-tag">' + esc(p.link_flair_text) + '</span>'
    : '';

  const style =
    '--sub-bg:'    + c.bg       + ';' +
    '--sub-text:'  + c.text     + ';' +
    '--sub-icon:'  + c.icon     + ';' +
    '--sub-bg-d:'  + c.darkBg   + ';' +
    '--sub-text-d:'+ c.darkText + ';' +
    '--sub-icon-d:'+ c.darkIcon;

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
          '<span class="meta-dot">·</span>' +
          '<span class="meta-auth">u/' + esc(p.author) + '</span>' +
          '<span class="meta-dot">·</span>' +
          '<time class="meta-time">' + ago(p.created_utc) + '</time>' +
          flair +
        '</div>' +
        '<h2 class="card-title">' +
          '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + esc(p.title) + '</a>' +
        '</h2>' +
        snippet +
        '<div class="card-stats">' +
          '<span class="stat score">▲ ' + fmt(p.score) + '</span>' +
          '<span class="stat comments">💬 ' + fmt(p.num_comments) + '</span>' +
          awards +
          '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer" class="view-btn">View on Reddit →</a>' +
        '</div>' +
      '</div>' +
    '</article>'
  );
}

// ── Knowledge panel ────────────────────────────────────────────────────────────

function renderKnowledge(post, comments, sub) {
  const topComment = comments.find(function (c) {
    return c.score > 3 && demd(c.body).length > 80 && c.author !== '[deleted]';
  });
  if (!topComment && !post.selftext) return null;

  const subName = esc(post.subreddit);
  const c       = subColors(post.subreddit);
  const postUrl = 'https://www.reddit.com' + post.permalink;
  const subUrl  = 'https://www.reddit.com/r/' + subName + '/';
  const members = sub && sub.subscribers ? fmt(sub.subscribers) + ' members' : '';
  const subDesc = sub && sub.public_description
    ? '<p class="k-sub-desc">' + esc(trunc(demd(sub.public_description), 120)) + '</p>'
    : '';

  const style =
    '--sub-bg:'    + c.bg       + ';' +
    '--sub-text:'  + c.text     + ';' +
    '--sub-icon:'  + c.icon     + ';' +
    '--sub-bg-d:'  + c.darkBg   + ';' +
    '--sub-text-d:'+ c.darkText + ';' +
    '--sub-icon-d:'+ c.darkIcon;

  let answer = '';
  if (topComment) {
    const body = esc(trunc(demd(topComment.body), 500));
    answer =
      '<div class="k-answer">' +
        '<div class="k-answer-meta">' +
          '<span class="k-answer-label">Top Answer</span>' +
          '<span class="k-answer-by">by ' +
            '<a href="https://www.reddit.com/u/' + esc(topComment.author) + '/" target="_blank" rel="noopener noreferrer">' +
              'u/' + esc(topComment.author) +
            '</a>' +
          '</span>' +
          '<span class="k-score">▲ ' + fmt(topComment.score) + '</span>' +
        '</div>' +
        '<p class="k-answer-body">' + body + '</p>' +
        '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-read-more">Read full discussion →</a>' +
      '</div>';
  } else {
    const body = esc(trunc(demd(post.selftext), 450));
    answer =
      '<div class="k-answer">' +
        '<p class="k-answer-body">' + body + '</p>' +
        '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-read-more">Read full post →</a>' +
      '</div>';
  }

  return (
    '<div class="k-header">' +
      '<a href="' + subUrl + '" target="_blank" rel="noopener noreferrer" class="k-sub" style="' + style + '">' +
        '<span class="k-sub-icon">' + initial(post.subreddit) + '</span>' +
        '<div class="k-sub-info">' +
          '<span class="k-sub-name">r/' + subName + '</span>' +
          (members ? '<span class="k-sub-members">' + members + '</span>' : '') +
        '</div>' +
      '</a>' +
      '<span class="k-tag">Featured Answer</span>' +
    '</div>' +
    subDesc +
    '<div class="k-post-title">' +
      '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer">' + esc(post.title) + '</a>' +
    '</div>' +
    answer +
    '<div class="k-actions">' +
      '<a href="' + esc(postUrl) + '" target="_blank" rel="noopener noreferrer" class="k-btn primary">View Discussion</a>' +
      '<a href="' + subUrl + '" target="_blank" rel="noopener noreferrer" class="k-btn">Browse r/' + subName + '</a>' +
    '</div>'
  );
}

async function buildKnowledge(topPost) {
  if (!topPost || topPost.score < 10) return;

  el.knowledgePanel.innerHTML =
    '<div class="k-skeleton">' +
      '<div class="skel-line" style="width:40%"></div>' +
      '<div class="skel-line" style="width:70%"></div>' +
      '<div class="skel-line" style="width:95%"></div>' +
      '<div class="skel-line" style="width:85%"></div>' +
      '<div class="skel-line" style="width:60%"></div>' +
    '</div>';
  el.knowledgePanel.classList.remove('hidden');

  try {
    const results = await Promise.allSettled([
      RedditAPI.getComments(topPost.subreddit, topPost.id),
      RedditAPI.getSubreddit(topPost.subreddit),
    ]);

    const comments = results[0].status === 'fulfilled' ? results[0].value.comments : [];
    const subData  = results[1].status === 'fulfilled' ? results[1].value       : null;

    const html = renderKnowledge(topPost, comments, subData);
    if (html) {
      el.knowledgePanel.innerHTML = html;
    } else {
      el.knowledgePanel.classList.add('hidden');
    }
  } catch (_) {
    el.knowledgePanel.classList.add('hidden');
  }
}

// ── Main search ────────────────────────────────────────────────────────────────

async function search(reset) {
  if (reset === undefined) reset = true;
  if (!state.query.trim()) return;

  showResults();
  setLoading(true);

  el.resultsInput.value  = state.query;
  el.subredditInput.value = state.subreddit;
  document.title = state.query + ' — RedditSearch';

  setPill('sort', state.sort);
  setPill('time', state.time);

  if (reset) {
    state.posts = [];
    state.after = null;
    el.resultsList.innerHTML = '';
    el.statsBar.textContent  = '';
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

    if (reset && res.posts.length > 0) {
      const top = res.posts[0];
      if (isQ(state.query) || top.score > 50) {
        buildKnowledge(top);
      }
    }
  } catch (err) {
    setLoading(false);
    if (err.status === 429) {
      showErr('Reddit is rate-limiting requests. Wait a moment and try again.');
    } else if (String(err.message || '').toLowerCase().includes('fetch')) {
      showErr('Could not reach Reddit. Check your internet connection.');
    } else {
      showErr(err.message || 'Unexpected error.');
    }
  }
}

// ── Init ───────────────────────────────────────────────────────────────────────

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
