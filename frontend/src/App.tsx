import { useState } from 'react';
import { DocumentQA } from './components/DocumentQA';
import { AIAssistant } from './components/AIAssistant';
import { FunTools } from './components/FunTools';
import { ImageStudio } from './components/ImageStudio';
import { MemoryPalace } from './components/MemoryPalace';

function App() {
  const [activeTab, setActiveTab] = useState('docqa');

  return (
    <div className="container">
      <header className="header">
        <h1>Nexus Mind</h1>
        <p className="subtitle">All Your AI Tools, One Intelligent Hub</p>
      </header>

      <nav className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'docqa' ? 'active' : ''}`}
          onClick={() => setActiveTab('docqa')}
        >
          Document Q&A
        </button>
        <button 
          className={`tab-btn ${activeTab === 'ai' ? 'active' : ''}`}
          onClick={() => setActiveTab('ai')}
        >
          AI Assistant
        </button>
        <button 
          className={`tab-btn ${activeTab === 'memory' ? 'active' : ''}`}
          onClick={() => setActiveTab('memory')}
        >
          Memory Palace
        </button>
        <button 
          className={`tab-btn ${activeTab === 'fun' ? 'active' : ''}`}
          onClick={() => setActiveTab('fun')}
        >
          Fun Tools
        </button>
        <button 
          className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
          onClick={() => setActiveTab('image')}
        >
          Image Studio
        </button>
      </nav>

      <main>
        {activeTab === 'docqa' && <DocumentQA />}
        {activeTab === 'ai' && <AIAssistant />}
        {activeTab === 'memory' && <MemoryPalace />}
        {activeTab === 'fun' && <FunTools />}
        {activeTab === 'image' && <ImageStudio />}
      </main>
      
      <footer style={{ position: 'relative' }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          width: '120px',
          height: '1px',
          background: 'linear-gradient(90deg, transparent, var(--accent-gold), transparent)',
          opacity: 0.4
        }}></div>
        <p style={{ 
          fontSize: '15px', 
          fontWeight: 500,
          letterSpacing: '0.3px',
          marginBottom: '12px'
        }}>
          Powered by <strong style={{ 
            color: 'var(--accent-gold)',
            fontFamily: 'var(--font-display)',
            fontWeight: 600
          }}>Nexus Mind</strong> • {new Date().getFullYear()}
        </p>
        <p style={{ 
          fontSize: '13px', 
          opacity: 0.5,
          fontWeight: 300,
          letterSpacing: '0.5px'
        }}>
          Built with LiteLLM, Ollama, LlamaIndex & FastAPI
        </p>
      </footer>
    </div>
  );
}

export default App;
