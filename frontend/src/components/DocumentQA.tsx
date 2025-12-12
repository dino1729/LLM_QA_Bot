import { useState, useEffect } from 'react';
import { api } from '../api';

export function DocumentQA() {
  const [subTab, setSubTab] = useState('video');
  const [provider, setProvider] = useState('litellm');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  
  // Inputs
  const [url, setUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  
  // Chat
  const [chatHistory, setChatHistory] = useState<string[][]>([]);
  const [message, setMessage] = useState('');
  const [answering, setAnswering] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Fetch models when provider changes
  useEffect(() => {
    let cancelled = false;
    let cancelFn: (() => void) | null = null;
    
    const fetchModels = async () => {
      setLoadingModels(true);
      const apiCall = api.models.list(provider);
      cancelFn = apiCall.cancel;
      
      try {
        const response = await apiCall.promise;
        if (!cancelled) {
          setAvailableModels(response.models || []);
          // Set first model as default
          if (response.models && response.models.length > 0) {
            setModel(response.models[0]);
          }
        }
      } catch (error) {
        // Ignore abort errors (component unmounted or request cancelled)
        if (error instanceof Error && (error as any).aborted) {
          return;
        }
        if (!cancelled) {
          console.error('Error fetching models:', error);
          setAvailableModels([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingModels(false);
        }
        cancelFn = null;
      }
    };
    
    fetchModels();
    
    return () => {
      cancelled = true;
      if (cancelFn) {
        cancelFn();
      }
    };
  }, [provider]);

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    setModel(''); // Reset model when provider changes
  };

  const handleProcess = async () => {
    if (!model) return;
    
    // Validate URL for video, article, and media tabs
    if ((subTab === 'video' || subTab === 'article' || subTab === 'media') && (!url || !url.trim())) {
      alert(`Please enter a valid ${subTab} URL`);
      return;
    }
    
    setLoading(true);
    setResult(null);
    let cancelFn: (() => void) | null = null;
    
    try {
      // Format model name as PROVIDER:model_name for backend
      const modelName = `${provider.toUpperCase()}:${model}`;
      let apiCall;
      if (subTab === 'video') {
        apiCall = api.analyze.youtube(url, false, modelName);
      } else if (subTab === 'article') {
        apiCall = api.analyze.article(url, false, modelName);
      } else if (subTab === 'media') {
        apiCall = api.analyze.media(url, false, modelName);
      } else if (subTab === 'file' && file) {
        const fd = new FormData();
        fd.append('files', file);
        fd.append('memorize', 'false');
        fd.append('model_name', modelName);
        apiCall = api.analyze.file(fd);
      } else {
        return;
      }
      
      cancelFn = apiCall.cancel;
      const res = await apiCall.promise;
      setResult(res);
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      console.error(e);
      alert('Error processing: ' + e);
    } finally {
      setLoading(false);
      cancelFn = null;
    }
  };

  const handleAsk = async () => {
    if (!message.trim() || !model) return;
    const newHistory = [...chatHistory, [message, '...']];
    setChatHistory(newHistory);
    setMessage('');
    setAnswering(true);
    
    try {
      // Format model name as PROVIDER:model_name for backend
      const modelName = `${provider.toUpperCase()}:${model}`;
      const { promise, cancel } = api.docqa.ask(message, chatHistory, modelName);
      const res = await promise;
      const updated = [...chatHistory, [message, res.answer]];
      setChatHistory(updated);
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      const updated = [...chatHistory, [message, 'Error: ' + e]];
      setChatHistory(updated);
    } finally {
      setAnswering(false);
    }
  };

  return (
    <div className="doc-qa">
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'nowrap' }}>
          <h2 className="card-title" style={{ margin: 0 }}>Analysis Configuration</h2>
          <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
            <select 
              value={provider} 
              onChange={e => handleProviderChange(e.target.value)}
              style={{ minWidth: '120px', fontSize: '14px' }}
            >
              <option value="litellm">LiteLLM</option>
              <option value="ollama">Ollama</option>
            </select>
            <select 
              value={model} 
              onChange={e => setModel(e.target.value)}
              style={{ minWidth: '180px', fontSize: '14px' }}
              disabled={loadingModels || availableModels.length === 0}
            >
              {loadingModels ? (
                <option value="">Loading models...</option>
              ) : availableModels.length === 0 ? (
                <option value="">No models available</option>
              ) : (
                availableModels.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))
              )}
            </select>
          </div>
        </div>
        
        <div style={{ marginBottom: '24px' }}>
          <div className="pill-tabs">
            {['video', 'article', 'file', 'media'].map(t => (
              <button 
                key={t}
                className={`pill-btn ${subTab === t ? 'active' : ''}`}
                onClick={() => setSubTab(t)}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="input-group" style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
          {subTab === 'file' ? (
            <div style={{ flex: 1 }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '14px 18px',
                background: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--border-radius-sm)',
                cursor: 'pointer',
                transition: 'all var(--transition-base)',
              }}
              onMouseOver={e => {
                e.currentTarget.style.borderColor = 'var(--accent-gold)';
                e.currentTarget.style.background = 'rgba(0, 0, 0, 0.4)';
              }}
              onMouseOut={e => {
                e.currentTarget.style.borderColor = 'var(--border-medium)';
                e.currentTarget.style.background = 'rgba(0, 0, 0, 0.3)';
              }}
              >
                <span style={{ fontSize: '20px' }}>📎</span>
                <span style={{ 
                  color: file ? 'var(--accent-gold)' : 'var(--text-muted)',
                  fontSize: '15px',
                  fontWeight: file ? 500 : 400
                }}>
                  {file ? file.name : 'Choose File'}
                </span>
                <input 
                  type="file" 
                  onChange={e => setFile(e.target.files?.[0] || null)}
                  style={{ display: 'none' }}
                />
              </label>
            </div>
          ) : (
            <input 
              type="text" 
              placeholder={`Enter ${subTab} URL...`}
              value={url}
              onChange={e => setUrl(e.target.value)}
              style={{ flex: 1 }}
            />
          )}
          <button className="btn btn-primary" onClick={handleProcess} disabled={loading} style={{ minWidth: '140px' }}>
            {loading ? (
              <>
                <div className="loader"></div>
                <span style={{ marginLeft: '8px' }}>Processing</span>
              </>
            ) : 'Process'}
          </button>
        </div>
        
        {result && (
          <div style={{ marginTop: '32px' }}>
            <div className="card" style={{ 
              background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(22, 25, 32, 0.8) 100%)', 
              borderColor: 'rgba(212, 175, 55, 0.2)' 
            }}>
              <h3 style={{ 
                marginTop: 0, 
                fontFamily: 'var(--font-display)', 
                color: 'var(--text-primary)',
                fontSize: '20px'
              }}>
                {result.video_title || result.article_title || result.file_title || result.media_title || 'Analysis Result'}
              </h3>
              <p style={{ color: 'var(--accent-gold)', marginBottom: '16px', fontSize: '14px' }}>{result.message}</p>
              <details open>
                <summary style={{ 
                  cursor: 'pointer', 
                  color: 'var(--text-secondary)', 
                  fontWeight: 600,
                  fontSize: '15px',
                  marginBottom: '12px',
                  userSelect: 'none'
                }}>
                  📋 Key Takeaways
                </summary>
                <div className="prose" style={{ 
                  marginTop: '16px', 
                  paddingLeft: '20px', 
                  borderLeft: '3px solid var(--accent-gold)',
                  paddingTop: '4px',
                  paddingBottom: '4px'
                }}>
                  {result.summary ?? 'No summary available'}
                </div>
              </details>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Ask Questions</h2>
          <button 
            className="btn btn-secondary" 
            disabled={resetting}
            onClick={async () => {
              setResetting(true);
              try {
                await api.docqa.reset().promise;
                setChatHistory([]);
                alert('Database reset successfully!');
              } catch (e) {
                if (e instanceof Error && (e as any).aborted) return;
                alert('Error resetting database: ' + (e instanceof Error ? e.message : String(e)));
              } finally {
                setResetting(false);
              }
            }}
          >
            {resetting ? (
              <>
                <div className="loader" style={{ width: '14px', height: '14px' }}></div>
                <span style={{ marginLeft: '6px' }}>Resetting...</span>
              </>
            ) : 'Reset DB'}
          </button>
        </div>
        
        <div className="chat-window" style={{ 
          height: '450px', 
          overflowY: 'auto', 
          marginBottom: '24px', 
          padding: '24px' 
        }}>
          {chatHistory.map((msg, i) => (
            <div key={i} style={{ marginBottom: '24px', animation: 'slideUp 0.3s ease-out' }}>
              <div style={{ textAlign: 'right', marginBottom: '8px' }}>
                <span style={{ 
                  background: 'linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-dark) 100%)', 
                  color: 'black', 
                  padding: '10px 18px', 
                  borderRadius: '18px 18px 2px 18px', 
                  display: 'inline-block', 
                  fontWeight: 500,
                  fontSize: '15px',
                  maxWidth: '80%',
                  boxShadow: '0 2px 8px var(--accent-glow)'
                }}>
                  {msg[0]}
                </span>
              </div>
              <div style={{ textAlign: 'left' }}>
                <div style={{ 
                  background: 'rgba(255,255,255,0.05)', 
                  padding: '16px 20px', 
                  borderRadius: '18px 18px 18px 2px', 
                  display: 'inline-block', 
                  maxWidth: '85%', 
                  lineHeight: 1.7,
                  fontSize: '15px',
                  border: '1px solid var(--border-subtle)'
                }}>
                  {msg[1]}
                </div>
              </div>
            </div>
          ))}
          {chatHistory.length === 0 && (
            <div className="empty-state">
              <p>Ask questions about your processed documents</p>
              <p style={{ fontSize: '13px', marginTop: '8px', opacity: 0.7 }}>
                Process a video, article, file, or media URL above to get started
              </p>
            </div>
          )}
        </div>
        
        <div style={{ display: 'flex', gap: '12px' }}>
          <input 
            type="text" 
            placeholder="Ask a question..." 
            value={message}
            onChange={e => setMessage(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleAsk()}
            style={{ flex: 1 }}
          />
          <button 
            className="btn btn-primary" 
            onClick={handleAsk} 
            disabled={answering || !result}
            style={{ minWidth: '100px' }}
          >
            {answering ? <div className="loader"></div> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
