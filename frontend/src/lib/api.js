/**
 * API client for the Personal Knowledge Assistant backend.
 */

const BASE_URL = '';

/**
 * Generic fetch wrapper with error handling.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`API Error ${res.status}: ${errBody}`);
    }
    return res;
  } catch (err) {
    if (err.message.startsWith('API Error')) throw err;
    throw new Error(`Network error: ${err.message}`);
  }
}

// ========== Health ==========
export async function getHealth() {
  const res = await apiFetch('/api/health');
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
  const res = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });

  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status}`);
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
