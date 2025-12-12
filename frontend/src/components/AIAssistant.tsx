import { useState, useEffect } from 'react';
import { api } from '../api';

export function AIAssistant() {
  const [provider, setProvider] = useState('litellm');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [maxTokens, setMaxTokens] = useState(4096);
  const [temperature, setTemperature] = useState(0.5);
  const [loadingModels, setLoadingModels] = useState(false);
  
  const [history, setHistory] = useState<string[][]>([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

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

  const handleSend = async () => {
    if (!message.trim() || !model) return;
    const userMessage = message; // Capture message before clearing
    setHistory(prev => [...prev, [userMessage, '...']]);
    setMessage('');
    setLoading(true);
    
    try {
      // Format model name as PROVIDER:model_name for backend
      const modelName = `${provider.toUpperCase()}:${model}`;
      const { promise } = api.chat.internet(userMessage, history, modelName, maxTokens, temperature);
      const res = await promise;
      // Replace placeholder with actual response using functional update
      setHistory(prev => [...prev.slice(0, -1), [userMessage, res.response]]);
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      // Replace placeholder with error message using functional update
      setHistory(prev => [...prev.slice(0, -1), [userMessage, 'Error: ' + e]]);
    } finally {
      setLoading(false);
    }
  };

  const exampleQueries = [
    "What's the latest news about AI?",
    "Research quantum computing breakthroughs",
    "What's the weather in San Francisco?"
  ];

  return (
    <div className="ai-assistant">
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'nowrap' }}>
          <h2 className="card-title" style={{ margin: 0 }}>Internet-Connected Assistant</h2>
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
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '20px', 
          marginBottom: '24px', 
          padding: '20px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: 'var(--border-radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div>
            <label style={{ 
              fontSize: '13px', 
              color: 'var(--text-secondary)', 
              fontWeight: 500,
              display: 'block',
              marginBottom: '12px'
            }}>
              Max Tokens: <span style={{ color: 'var(--accent-gold)' }}>{maxTokens}</span>
            </label>
            <input 
              type="range" min="100" max="8000" step="100"
              value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))}
            />
          </div>
          <div>
            <label style={{ 
              fontSize: '13px', 
              color: 'var(--text-secondary)', 
              fontWeight: 500,
              display: 'block',
              marginBottom: '12px'
            }}>
              Temperature: <span style={{ color: 'var(--accent-gold)' }}>{temperature.toFixed(1)}</span>
            </label>
            <input 
              type="range" min="0" max="1" step="0.1"
              value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="chat-window" style={{ 
          height: '500px', 
          overflowY: 'auto', 
          marginBottom: '24px', 
          padding: '28px'
        }}>
          {history.map((msg, i) => (
            <div key={i} style={{ marginBottom: '28px', animation: 'slideUp 0.3s ease-out' }}>
              <div style={{ textAlign: 'right', marginBottom: '10px' }}>
                <span style={{ 
                  background: 'linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-dark) 100%)', 
                  color: 'black', 
                  padding: '12px 20px', 
                  borderRadius: '20px 20px 2px 20px', 
                  display: 'inline-block',
                  fontWeight: 500,
                  fontSize: '15px',
                  maxWidth: '75%',
                  boxShadow: '0 2px 8px var(--accent-glow)',
                  lineHeight: 1.5
                }}>
                  {msg[0]}
                </span>
              </div>
              <div style={{ textAlign: 'left' }}>
                <div style={{ 
                  background: 'rgba(255,255,255,0.05)', 
                  padding: '16px 20px', 
                  borderRadius: '20px 20px 20px 2px', 
                  display: 'inline-block', 
                  maxWidth: '85%', 
                  lineHeight: 1.7,
                  fontSize: '15px',
                  border: '1px solid var(--border-subtle)',
                  whiteSpace: 'pre-wrap'
                }}>
                  <div className="prose">{msg[1]}</div>
                </div>
              </div>
            </div>
          ))}
          {history.length === 0 && (
            <div className="empty-state">
              <p style={{ marginBottom: '20px', fontSize: '16px' }}>Start a conversation with your AI assistant</p>
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '8px',
                maxWidth: '400px',
                margin: '0 auto'
              }}>
                <p style={{ fontSize: '12px', opacity: 0.7, marginBottom: '8px' }}>Try asking:</p>
                {exampleQueries.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setMessage(q)}
                    style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-secondary)',
                      padding: '10px 16px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      textAlign: 'left',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={e => {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                      e.currentTarget.style.borderColor = 'var(--accent-gold)';
                    }}
                    onMouseOut={e => {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                      e.currentTarget.style.borderColor = 'var(--border-subtle)';
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <input 
            type="text" 
            placeholder="Ask anything (e.g. 'Latest AI news')..." 
            value={message}
            onChange={e => setMessage(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
            style={{ flex: 1 }}
          />
          <button 
            className="btn btn-primary" 
            onClick={handleSend} 
            disabled={loading}
            style={{ minWidth: '100px' }}
          >
            {loading ? <div className="loader"></div> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
