import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { BrainCircuit, Activity } from 'lucide-react';
import { 
  getHealth, 
  getDocuments, 
  getStats, 
  uploadDocuments, 
  ingestUrl, 
  deleteDocument, 
  streamChat, 
  clearChatHistory,
  login,
  register,
  getProfile
} from './lib/api';

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('ka-theme') || 'dark');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Auth State
  const [token, setToken] = useState(() => localStorage.getItem('ka-token'));
  const [currentUser, setCurrentUser] = useState(null);
  const [authMode, setAuthMode] = useState('login'); // 'login' | 'register'
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  // Theme persistence
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ka-theme', theme);
  }, [theme]);

  // Load profile when token changes
  useEffect(() => {
    if (token) {
      loadProfile();
    } else {
      setCurrentUser(null);
    }
  }, [token]);

  // Load initial data only if authenticated
  useEffect(() => {
    if (token && currentUser) {
      loadHealth();
      loadDocuments();
      loadStats();
    }
  }, [token, currentUser]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        document.getElementById('chat-input')?.focus();
      }
      if (e.key === 'b' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSidebarOpen(s => !s);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Listen for unauthorized events
  useEffect(() => {
    const handleUnauthorized = () => {
      setToken(null);
      setCurrentUser(null);
      addToast('Session Expired', 'Please log in again.', 'error');
    };
    window.addEventListener('auth-unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth-unauthorized', handleUnauthorized);
  }, []);

  function addToast(title, desc, type = 'info') {
    const id = Date.now();
    setToasts(t => [...t, { id, title, desc, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }

  async function loadProfile() {
    try {
      const user = await getProfile();
      setCurrentUser(user);
    } catch {
      setToken(null);
      localStorage.removeItem('ka-token');
    }
  }

  async function loadHealth() {
    try {
      const data = await getHealth();
      setHealth(data);
    } catch { setHealth(null); }
  }

  async function loadDocuments() {
    try {
      const data = await getDocuments();
      setDocuments(data);
    } catch { /* silent */ }
  }

  async function loadStats() {
    try {
      const data = await getStats();
      setStats(data);
    } catch { /* silent */ }
  }

  async function handleAuthSubmit(e) {
    e.preventDefault();
    if (!authUsername.trim() || !authPassword.trim()) {
      setAuthError('All fields are required');
      return;
    }
    setAuthError('');
    setAuthLoading(true);
    try {
      if (authMode === 'login') {
        const data = await login(authUsername.trim(), authPassword.trim());
        setToken(data.access_token);
        addToast('Welcome back!', `Logged in as ${authUsername.trim()}`);
      } else {
        await register(authUsername.trim(), authPassword.trim());
        addToast('Registration Complete', 'You can now log in with your credentials.');
        setAuthMode('login');
        setAuthPassword('');
      }
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem('ka-token');
    setToken(null);
    setCurrentUser(null);
    setMessages([]);
    setDocuments([]);
    setStats(null);
    setConversationId(null);
    setAuthUsername('');
    setAuthPassword('');
    addToast('Logged Out', 'Successfully logged out.');
  }

  async function handleUpload(files) {
    setUploading(true);
    try {
      const result = await uploadDocuments(files);
      addToast('Upload Complete', `Processed ${result.processed} document(s).`);
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('Upload Failed', err.message, 'error');
    } finally {
      setUploading(false);
    }
  }

  async function handleUrlIngest(url) {
    try {
      await ingestUrl(url);
      addToast('URL Ingested', 'Content has been processed.');
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('URL Failed', err.message, 'error');
    }
  }

  async function handleDeleteDoc(docId) {
    try {
      await deleteDocument(docId);
      loadDocuments();
      loadStats();
    } catch (err) {
      addToast('Delete Failed', err.message, 'error');
    }
  }

  async function handleSendMessage(content) {
    const userMsg = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    const assistantMsgId = crypto.randomUUID();
    const assistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      sources: [],
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsTyping(true);

    try {
      for await (const event of streamChat(content, conversationId)) {
        if (event.event === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + event.data.token }
                : m
            )
          );
        } else if (event.event === 'sources') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, sources: event.data.sources }
                : m
            )
          );
        } else if (event.event === 'done') {
          setConversationId(event.data.conversation_id);
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId ? { ...m, isStreaming: false } : m
            )
          );
        } else if (event.event === 'error') {
          throw new Error(event.data.error);
        }
      }
    } catch (err) {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsgId
            ? { ...m, isStreaming: false, content: m.content || 'Sorry, an error occurred.' }
            : m
        )
      );
      addToast('Chat Error', err.message, 'error');
    } finally {
      setIsTyping(false);
    }
  }

  async function handleClearChat() {
    try {
      await clearChatHistory();
      setMessages([]);
      setConversationId(null);
      loadDocuments();
      loadStats();
      addToast('Chat Cleared', 'Conversation history and documents removed.');
    } catch (err) {
      addToast('Action Failed', err.message, 'error');
    }
  }

  // Render Login overlay if unauthenticated
  if (!token || !currentUser) {
    return (
      <div className="auth-layout">
        <div className="auth-container">
          <div className="auth-card">
            <div className="auth-header">
              <div className="auth-logo">
                <BrainCircuit size={24} />
              </div>
              <h2 className="auth-title">
                {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
              </h2>
              <p className="auth-subtitle">
                {authMode === 'login' 
                  ? 'Log in to your Personal Knowledge Assistant' 
                  : 'Register to start ingesting knowledge'}
              </p>
            </div>
            
            <form onSubmit={handleAuthSubmit} className="auth-form">
              {authError && (
                <div className="auth-error">
                  <Activity size={14} className="spinner" style={{ color: 'var(--destructive)' }} />
                  <span>{authError}</span>
                </div>
              )}
              
              <div className="auth-group">
                <label className="auth-label">Username</label>
                <div className="auth-input-wrapper">
                  <input
                    type="text"
                    className="auth-input"
                    placeholder="Enter username"
                    value={authUsername}
                    onChange={e => setAuthUsername(e.target.value)}
                    disabled={authLoading}
                    autoFocus
                  />
                </div>
              </div>
              
              <div className="auth-group">
                <label className="auth-label">Password</label>
                <div className="auth-input-wrapper">
                  <input
                    type="password"
                    className="auth-input"
                    placeholder="Enter password"
                    value={authPassword}
                    onChange={e => setAuthPassword(e.target.value)}
                    disabled={authLoading}
                  />
                </div>
              </div>
              
              <button type="submit" className="auth-submit-btn" disabled={authLoading}>
                {authLoading ? 'Please wait...' : (authMode === 'login' ? 'Sign In' : 'Sign Up')}
              </button>
            </form>
            
            <div className="auth-switch">
              {authMode === 'login' ? "Don't have an account?" : "Already have an account?"}
              <button
                onClick={() => {
                  setAuthMode(authMode === 'login' ? 'register' : 'login');
                  setAuthError('');
                }}
                className="auth-switch-link"
                disabled={authLoading}
              >
                {authMode === 'login' ? 'Sign Up' : 'Sign In'}
              </button>
            </div>
          </div>
        </div>

        {/* Toasts */}
        <div className="toast-container">
          {toasts.map(t => (
            <div key={t.id} className={`toast ${t.type === 'error' ? 'error' : ''}`}>
              <div className="toast-title">{t.title}</div>
              {t.desc && <div className="toast-desc">{t.desc}</div>}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      <Sidebar
        open={sidebarOpen}
        documents={documents}
        stats={stats}
        uploading={uploading}
        theme={theme}
        currentUser={currentUser}
        onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
        onUpload={handleUpload}
        onUrlIngest={handleUrlIngest}
        onDeleteDoc={handleDeleteDoc}
        onClearChat={handleClearChat}
        onLogout={handleLogout}
      />
      <ChatArea
        messages={messages}
        isTyping={isTyping}
        health={health}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(s => !s)}
        onSendMessage={handleSendMessage}
      />

      {/* Toasts */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type === 'error' ? 'error' : ''}`}>
            <div className="toast-title">{t.title}</div>
            {t.desc && <div className="toast-desc">{t.desc}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
