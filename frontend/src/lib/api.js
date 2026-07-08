/**
 * API client for the Personal Knowledge Assistant backend.
 */

const BASE_URL = '';

/**
 * Generic fetch wrapper with error handling and JWT support.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const token = localStorage.getItem('ka-token');
  
  const headers = { ...options.headers };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
      localStorage.removeItem('ka-token');
      window.dispatchEvent(new Event('auth-unauthorized'));
      throw new Error('Unauthorized');
    }
    if (!res.ok) {
      const errBody = await res.text();
      let detail = `API Error ${res.status}`;
      try {
        const parsed = JSON.parse(errBody);
        detail = parsed.detail || detail;
      } catch {
        if (errBody) detail = errBody;
      }
      throw new Error(detail);
    }
    return res;
  } catch (err) {
    if (err.message === 'Unauthorized' || err.message.startsWith('API Error') || err.message.includes('Error')) throw err;
    throw new Error(`Network error: ${err.message}`);
  }
}

// ========== Auth ==========
export async function login(username, password) {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);
  
  const res = await fetch(`${BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString()
  });
  
  if (!res.ok) {
    const errBody = await res.text();
    let detail = 'Incorrect username or password';
    try {
      const parsed = JSON.parse(errBody);
      detail = parsed.detail || detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  
  const data = await res.json();
  localStorage.setItem('ka-token', data.access_token);
  return data;
}

export async function register(username, password) {
  const res = await fetch(`${BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  
  if (!res.ok) {
    const errBody = await res.text();
    let detail = 'Registration failed';
    try {
      const parsed = JSON.parse(errBody);
      detail = parsed.detail || detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  
  return res.json();
}

export async function getProfile() {
  const res = await apiFetch('/api/auth/me');
  return res.json();
}

// ========== Health (Public) ==========
export async function getHealth() {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ========== Stats ==========
export async function getStats() {
  const res = await apiFetch('/api/stats');
  return res.json();
}

// ========== Documents ==========
export async function getDocuments() {
  const res = await apiFetch('/api/documents');
  return res.json();
}

export async function uploadDocuments(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await apiFetch('/api/documents/upload', {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

export async function ingestUrl(url) {
  const res = await apiFetch('/api/documents/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  return res.json();
}

export async function deleteDocument(docId) {
  const res = await apiFetch(`/api/documents/${docId}`, { method: 'DELETE' });
  return res.json();
}

// ========== Chat (SSE Streaming) ==========
/**
 * Send a chat message and return an async iterator of SSE events.
 * Each yielded item: { event: 'token'|'sources'|'done'|'error', data: object }
 */
export async function* streamChat(message, conversationId = null) {
  const token = localStorage.getItem('ka-token');
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });

  if (res.status === 401) {
    localStorage.removeItem('ka-token');
    window.dispatchEvent(new Event('auth-unauthorized'));
    throw new Error('Unauthorized');
  }

  if (!res.ok) {
    const errBody = await res.text();
    let detail = `Chat API error: ${res.status}`;
    try {
      const parsed = JSON.parse(errBody);
      detail = parsed.detail || detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // Keep incomplete line in buffer

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.substring(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.substring(6).trim();
        if (!dataStr) continue;
        try {
          const data = JSON.parse(dataStr);
          yield { event: currentEvent, data };
        } catch {
          // Skip malformed JSON
        }
      }
    }
  }
}

export async function clearChatHistory() {
  const res = await apiFetch('/api/chat/history', { method: 'DELETE' });
  return res.json();
}
