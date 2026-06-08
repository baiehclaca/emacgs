const RedditAPI = (() => {
  const BASE = 'https://www.reddit.com';

  async function get(url) {
    const sep = url.includes('?') ? '&' : '?';
    const res = await fetch(url + sep + 'raw_json=1');
    if (!res.ok) {
      const e = new Error(`Reddit API ${res.status}`);
      e.status = res.status;
      throw e;
    }
    return res.json();
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
      posts: data.data.children.map(c => c.data),
      after: data.data.after,
    };
  }

  async function getComments(subreddit, postId) {
    const url = `${BASE}/r/${subreddit}/comments/${postId}.json?limit=15&sort=confidence&depth=1`;
    const [postData, commentData] = await get(url);
    const post = postData.data.children[0]?.data ?? null;
    const comments = commentData.data.children
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
