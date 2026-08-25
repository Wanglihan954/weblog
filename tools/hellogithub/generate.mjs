import { access, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HELLOGITHUB_API = 'https://api.hellogithub.com/v1/periodical/volume/';
const ASSETS_VERSION = 2;
const CONTENT_VERSION = 1;
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDirectory, '..', '..');
const dataPath = path.join(projectRoot, 'source', '_data', 'hellogithub.json');
const imageDirectory = path.join(projectRoot, 'source', 'images', 'hellogithub');
const avatarDirectory = path.join(projectRoot, 'source', 'images', 'github-avatars');

async function readJson(filePath, fallback) {
  try {
    return JSON.parse((await readFile(filePath, 'utf8')).replace(/^\uFEFF/, ''));
  } catch (error) {
    if (error.code === 'ENOENT') return fallback;
    throw error;
  }
}

async function writeJsonAtomic(filePath, data) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
  await rename(temporaryPath, filePath);
}

function escapeActionsMessage(value) {
  return String(value).replaceAll('%', '%25').replaceAll('\r', '%0D').replaceAll('\n', '%0A');
}

export function imageFilenameFromUrl(value) {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.hostname !== 'img.hellogithub.com') return null;
    const filename = path.posix.basename(url.pathname).split('!', 1)[0];
    return /^[a-zA-Z0-9_-]+\.(?:png|jpe?g|gif|webp)$/i.test(filename) ? filename : null;
  } catch {
    return null;
  }
}

export function shortDescription(value) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  const sentence = text.match(/^.*?[。！？!?](?=\s|$|[^。！？!?])/u);
  return (sentence ? sentence[0] : text).slice(0, 120);
}

export function fullDescription(value) {
  return String(value || '').replace(/\r\n?/g, '\n').trim();
}

export function avatarFilenameFromRepo(repo) {
  const owner = String(repo || '').split('/', 1)[0];
  return /^[a-zA-Z0-9-]{1,39}$/.test(owner) ? `${owner.toLowerCase()}.png` : null;
}

export async function fetchGitHubAvatarUrl(repo, fetchImpl = fetch, token = process.env.GITHUB_TOKEN || '') {
  const parts = String(repo || '').split('/');
  if (parts.length !== 2 || !parts.every((part) => /^[a-zA-Z0-9._-]+$/.test(part))) {
    throw new Error(`GitHub 仓库名称无效：${repo}`);
  }
  const headers = {
    Accept: 'application/vnd.github+json',
    'User-Agent': 'seventh-knot-hellogithub-sync/1.0',
    'X-GitHub-Api-Version': '2022-11-28'
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  const apiUrl = `https://api.github.com/repos/${parts.map(encodeURIComponent).join('/')}`;
  const response = await fetchImpl(apiUrl, {
    headers,
    signal: AbortSignal.timeout(15_000)
  });
  if (!response.ok) throw new Error(`GitHub 仓库 API 请求失败：${response.status} ${repo}`);
  const avatarUrl = (await response.json())?.owner?.avatar_url;
  try {
    const parsed = new URL(avatarUrl);
    if (parsed.protocol !== 'https:' || parsed.hostname !== 'avatars.githubusercontent.com') throw new Error();
    parsed.searchParams.set('s', '96');
    return parsed.toString();
  } catch {
    throw new Error(`GitHub 仓库未返回有效头像：${repo}`);
  }
}

function toProject(item, volume, availableAssets) {
  if (!item?.name || !item?.full_name || !item?.github_url || !item?.description) {
    throw new Error(`HelloGitHub 第 ${volume} 期包含字段不完整的项目`);
  }
  const imageFilename = imageFilenameFromUrl(item.image_url);
  const avatarFilename = avatarFilenameFromRepo(item.full_name);
  const coverAvailable = imageFilename && (!availableAssets || availableAssets.covers.has(imageFilename));
  const avatarAvailable = avatarFilename && (!availableAssets || availableAssets.avatars.has(avatarFilename));
  return {
    name: item.name,
    repo: item.full_name,
    desc: shortDescription(item.description),
    full_desc: fullDescription(item.description),
    url: item.github_url,
    cover: coverAvailable ? `/images/hellogithub/${imageFilename}` : null,
    avatar: avatarAvailable ? `/images/github-avatars/${avatarFilename}` : null,
    stars: Number(item.stars || 0),
    forks: Number(item.forks || 0),
    slug: item.rid || item.full_name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  };
}

export function buildFeed(payload, availableAssets = null) {
  if (!payload?.success || !Number.isInteger(payload.current_num) || !Array.isArray(payload.data)) {
    throw new Error('HelloGitHub API 未返回有效的最新月刊');
  }
  const volume = payload.current_num;
  const groups = payload.data.map((group) => ({
    key: `hg-${group.category_id}`,
    name: group.category_name,
    projects: (group.items || []).map((item) => toProject(item, volume, availableAssets))
  })).filter((group) => group.projects.length);
  const projectCount = groups.reduce((total, group) => total + group.projects.length, 0);
  if (!projectCount) throw new Error(`HelloGitHub 第 ${volume} 期没有可发布的项目`);
  return {
    assets_version: ASSETS_VERSION,
    content_version: CONTENT_VERSION,
    issue: volume,
    published_at: payload.publish_at || new Date().toISOString(),
    source_url: `https://hellogithub.com/periodical/volume/${volume}`,
    groups
  };
}

function assetJobs(payload) {
  const jobs = payload.data.flatMap((group) => (group.items || []).flatMap((item) => {
    const filename = imageFilenameFromUrl(item.image_url);
    const avatarFilename = avatarFilenameFromRepo(item.full_name);
    return [
      filename ? {
        kind: 'cover',
        url: item.image_url,
        filename,
        directory: imageDirectory,
        required: true,
        attempts: 3,
        timeoutMs: 30_000,
        headers: { Referer: 'https://hellogithub.com/' }
      } : null,
      avatarFilename ? {
        kind: 'avatar',
        repo: item.full_name,
        filename: avatarFilename,
        directory: avatarDirectory,
        required: false,
        attempts: 2,
        timeoutMs: 15_000,
        headers: { Accept: 'image/*' }
      } : null
    ].filter(Boolean);
  }));
  return [...new Map(jobs.map((job) => [`${job.kind}:${job.filename}`, job])).values()];
}

export async function fetchImageBytes(url, extraHeaders = {}, fetchImpl = fetch, timeoutMs = 30_000) {
  const response = await fetchImpl(url, {
    headers: {
      'User-Agent': 'seventh-knot-hellogithub-sync/1.0',
      ...extraHeaders
    },
    signal: AbortSignal.timeout(timeoutMs)
  });
  if (!response.ok) throw new Error(`图片下载失败：${response.status} ${url}`);
  const contentType = response.headers.get('content-type') || '';
  if (!/^image\/(?:png|jpeg|gif|webp)(?:;|$)/i.test(contentType)) {
    throw new Error(`图片类型无效：${contentType || 'unknown'} ${url}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  if (!bytes.length || bytes.length > MAX_IMAGE_BYTES) {
    throw new Error(`图片大小无效：${bytes.length} bytes ${url}`);
  }
  return bytes;
}

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function downloadAssets(payload, fetchImpl) {
  const available = { covers: new Set(), avatars: new Set() };
  await mkdir(imageDirectory, { recursive: true });
  await mkdir(avatarDirectory, { recursive: true });
  const jobs = assetJobs(payload);
  let nextJobIndex = 0;

  async function processNextJob() {
    while (nextJobIndex < jobs.length) {
      const job = jobs[nextJobIndex];
      nextJobIndex += 1;
      const targetPath = path.join(job.directory, job.filename);
      if (await fileExists(targetPath)) {
        available[job.kind === 'cover' ? 'covers' : 'avatars'].add(job.filename);
        continue;
      }

      let lastError;
      for (let attempt = 1; attempt <= job.attempts; attempt += 1) {
        try {
          const assetUrl = job.kind === 'avatar'
            ? await fetchGitHubAvatarUrl(job.repo, fetchImpl)
            : job.url;
          const bytes = await fetchImageBytes(assetUrl, job.headers, fetchImpl, job.timeoutMs);
          const temporaryPath = `${targetPath}.tmp`;
          await writeFile(temporaryPath, bytes);
          await rename(temporaryPath, targetPath);
          available[job.kind === 'cover' ? 'covers' : 'avatars'].add(job.filename);
          lastError = null;
          break;
        } catch (error) {
          lastError = error;
          if (attempt < job.attempts) await new Promise((resolve) => setTimeout(resolve, attempt * 500));
        }
      }
      if (lastError && job.required) throw lastError;
      if (lastError) console.warn(`[hellogithub] 头像下载失败，使用 GH 占位符：${job.repo}`);
    }
  }
  await Promise.all(Array.from({ length: Math.min(6, jobs.length) }, () => processNextJob()));
  return available;
}

function availableAssetsFromFeed(feed) {
  const available = { covers: new Set(), avatars: new Set() };
  for (const group of feed?.groups || []) {
    for (const project of group.projects || []) {
      if (project.cover?.startsWith('/images/hellogithub/')) {
        available.covers.add(path.posix.basename(project.cover));
      }
      if (project.avatar?.startsWith('/images/github-avatars/')) {
        available.avatars.add(path.posix.basename(project.avatar));
      }
    }
  }
  return available;
}

export async function fetchLatestPayload(fetchImpl = fetch) {
  const response = await fetchImpl(HELLOGITHUB_API, {
    headers: {
      Accept: 'application/json',
      'User-Agent': 'seventh-knot-hellogithub-sync/1.0'
    },
    signal: AbortSignal.timeout(30_000)
  });
  if (!response.ok) throw new Error(`HelloGitHub API 请求失败：${response.status} ${await response.text()}`);
  return response.json();
}

export async function syncLatest({ fetchImpl = fetch } = {}) {
  const current = await readJson(dataPath, null);
  const payload = await fetchLatestPayload(fetchImpl);
  const sameIssue = current?.issue === payload.current_num;
  const assetsCurrent = sameIssue && current?.assets_version === ASSETS_VERSION;
  const contentCurrent = sameIssue && current?.content_version === CONTENT_VERSION;
  if (assetsCurrent && contentCurrent) {
    console.log(`[hellogithub] 当前已是最新的第 ${payload.current_num} 期，图片也已本地化，无需更新。`);
    return current;
  }
  const availableAssets = assetsCurrent
    ? availableAssetsFromFeed(current)
    : await downloadAssets(payload, fetchImpl);
  const feed = buildFeed(payload, availableAssets);
  await writeJsonAtomic(dataPath, feed);
  const projectCount = feed.groups.reduce((total, group) => total + group.projects.length, 0);
  console.log(`[hellogithub] 已同步第 ${feed.issue} 期，共 ${projectCount} 个项目；未调用任何 AI 模型。`);
  return feed;
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (invokedPath === fileURLToPath(import.meta.url)) {
  syncLatest().catch((error) => {
    const detail = error.stack || error.message || String(error);
    console.error(`[hellogithub] 同步失败：${detail}`);
    if (process.env.GITHUB_ACTIONS === 'true') {
      console.error(`::error title=HelloGitHub 同步失败::${escapeActionsMessage(detail)}`);
    }
    process.exitCode = 1;
  });
}
