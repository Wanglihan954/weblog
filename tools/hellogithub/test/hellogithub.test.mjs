import test from 'node:test';
import assert from 'node:assert/strict';
import {
  avatarFilenameFromRepo,
  buildFeed,
  fetchGitHubAvatarUrl,
  fetchImageBytes,
  fullDescription,
  imageFilenameFromUrl,
  shortDescription
} from '../generate.mjs';

const payload = {
  success: true,
  current_num: 125,
  publish_at: '2026-08-28T08:00:00',
  data: [{
    category_id: 1,
    category_name: 'Python 项目',
    items: [{
      rid: 'demo-id',
      name: 'demo',
      full_name: 'owner/demo',
      github_url: 'https://github.com/owner/demo',
      description: '第一句介绍。第二句不会进入卡片摘要。',
      image_url: 'https://img.hellogithub.com/demo_123.png!small',
      stars: 1234,
      forks: 56
    }]
  }]
};

test('buildFeed creates the data shape used by the projects page', () => {
  const feed = buildFeed(payload);
  assert.equal(feed.issue, 125);
  assert.equal(feed.assets_version, 2);
  assert.equal(feed.content_version, 1);
  assert.equal(feed.source_url, 'https://hellogithub.com/periodical/volume/125');
  assert.equal(feed.groups[0].key, 'hg-1');
  assert.deepEqual(feed.groups[0].projects[0], {
    name: 'demo',
    repo: 'owner/demo',
    desc: '第一句介绍。',
    full_desc: '第一句介绍。第二句不会进入卡片摘要。',
    url: 'https://github.com/owner/demo',
    cover: '/images/hellogithub/demo_123.png',
    avatar: '/images/github-avatars/owner.png',
    stars: 1234,
    forks: 56,
    slug: 'demo-id'
  });
});

test('avatarFilenameFromRepo creates a safe local filename', () => {
  assert.equal(avatarFilenameFromRepo('OpenAI/demo'), 'openai.png');
  assert.equal(avatarFilenameFromRepo('../unsafe'), null);
});

test('fetchGitHubAvatarUrl uses the repository API and returns the owner avatar', async () => {
  let requestedUrl;
  let requestedOptions;
  const avatarUrl = await fetchGitHubAvatarUrl('OpenAI/demo', async (url, options) => {
    requestedUrl = url;
    requestedOptions = options;
    return new Response(JSON.stringify({
      owner: { avatar_url: 'https://avatars.githubusercontent.com/u/14957082?v=4' }
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }, 'test-token');
  assert.equal(requestedUrl, 'https://api.github.com/repos/OpenAI/demo');
  assert.equal(requestedOptions.headers.Authorization, 'Bearer test-token');
  assert.match(avatarUrl, /^https:\/\/avatars\.githubusercontent\.com\/u\/14957082\?/);
  assert.match(avatarUrl, /s=96/);
});

test('fetchImageBytes sends the anti-hotlink referer and validates the image', async () => {
  let requestOptions;
  const bytes = await fetchImageBytes(
    'https://img.hellogithub.com/demo.png',
    { Referer: 'https://hellogithub.com/' },
    async (_url, options) => {
      requestOptions = options;
      return new Response(Uint8Array.from([1, 2, 3]), {
        status: 200,
        headers: { 'Content-Type': 'image/png' }
      });
    }
  );
  assert.equal(requestOptions.headers.Referer, 'https://hellogithub.com/');
  assert.deepEqual([...bytes], [1, 2, 3]);
});

test('fetchImageBytes rejects non-image responses', async () => {
  await assert.rejects(
    fetchImageBytes('https://img.hellogithub.com/demo.png', {}, async () => (
      new Response('blocked', { status: 200, headers: { 'Content-Type': 'text/plain' } })
    )),
    /图片类型无效/
  );
});

test('imageFilenameFromUrl only accepts safe HelloGitHub image names', () => {
  assert.equal(imageFilenameFromUrl('https://img.hellogithub.com/a-b_1.webp!small'), 'a-b_1.webp');
  assert.equal(imageFilenameFromUrl('https://example.com/a.png'), null);
  assert.equal(imageFilenameFromUrl('https://img.hellogithub.com/not-a-picture.svg'), null);
});

test('shortDescription keeps a compact first sentence', () => {
  assert.equal(shortDescription('  简短介绍。 后续细节。 '), '简短介绍。');
  assert.equal(shortDescription('No punctuation'), 'No punctuation');
});

test('fullDescription preserves the complete HelloGitHub copy', () => {
  assert.equal(fullDescription('第一段。\r\n第二段。'), '第一段。\n第二段。');
});

test('invalid payload is rejected', () => {
  assert.throws(() => buildFeed({ success: false }), /未返回有效/);
});
