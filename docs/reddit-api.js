const RedditAPI = (() => {
  const BASE = 'https://www.reddit.com';

  async function get(url) {
    const sep = url.includes('?') ? '&' : '?';
    let res;
    try {
      res = await fetch(url + sep + 'raw_json=1');
    } catch (netErr) {
      const e = new Error('Could not connect to Reddit — check your internet connection.');
      e.status = 0;
      throw e;
    }

    if (!res.ok) {
      const e = new Error('Reddit returned HTTP ' + res.status);
      e.status = res.status;
      throw e;
    }

    // Reddit sometimes returns an HTML redirect page (200 OK but not JSON).
    // Detect this before calling .json() to avoid a confusing SyntaxError.
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('html')) {
      const e = new Error('Reddit returned an HTML page instead of data — you may need to open reddit.com in a new tab and solve a CAPTCHA, then try again.');
      e.status = res.status;
      throw e;
    }

    let data;
    try {
      data = await res.json();
    } catch (_) {
      const e = new Error('Reddit returned malformed data. Please try again.');
      e.status = res.status;
      throw e;
    }

    return data;
  }

  async function search({ query, sort = 'relevance', time = 'all', limit = 25, after = null, subreddit = null } = {}) {
    const params = new URLSearchParams({
      q: query,
      sort,
      t: time,
      limit: String(limit),
      include_over_18: 'off',
      type: 'link',
    });
    if (after) params.set('after', after);

    const base = subreddit
      ? `${BASE}/r/${encodeURIComponent(subreddit)}/search.json?restrict_sr=1&${params}`
      : `${BASE}/search.json?${params}`;

    const data = await get(base);
    return {
      posts: (data.data.children || []).map(c => c.data),
      after: data.data.after || null,
    };
  }

  async function getComments(subreddit, postId) {
    const url = `${BASE}/r/${subreddit}/comments/${postId}.json?limit=15&sort=confidence&depth=1`;
    const [postData, commentData] = await get(url);
    const post = postData.data.children[0]?.data ?? null;
    const comments = (commentData.data.children || [])
      .filter(c => c.kind === 't1' && !c.data.stickied)
      .map(c => c.data)
      .sort((a, b) => b.score - a.score);
    return { post, comments };
  }

  async function getSubreddit(name) {
    const data = await get(`${BASE}/r/${name}/about.json`);
    return data.data;
  }

  return { search, getComments, getSubreddit };
})();
