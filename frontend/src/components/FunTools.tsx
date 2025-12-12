import { useState, useEffect } from 'react';
import { api } from '../api';

export function FunTools() {
  const [subTab, setSubTab] = useState('city');
  const [provider, setProvider] = useState('litellm');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  
  // City planner
  const [city, setCity] = useState('Tokyo');
  const [days, setDays] = useState(3);
  const [cityResult, setCityResult] = useState('');
  const [cityLoading, setCityLoading] = useState(false);
  const [cityError, setCityError] = useState<string | null>(null);
  
  // Cravings
  const [craving, setCraving] = useState('');
  const [cravingResult, setCravingResult] = useState('');
  const [cravingLoading, setCravingLoading] = useState(false);
  const [cravingError, setCravingError] = useState<string | null>(null);

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

  const cityExamples = ['Tokyo', 'Paris', 'New York', 'Kyoto', 'Barcelona', 'London'];

  const handleCityPlan = async () => {
    // Validate model is selected
    if (!model) return;
    
    // Validate city input - trim and check if empty
    const trimmedCity = city.trim();
    if (!trimmedCity) {
      setCityError('Please enter a city name');
      return;
    }
    
    // Proceed with API call after validation
    setCityLoading(true);
    setCityResult('');
    setCityError(null); // Clear previous errors
    try {
      // Format model name as PROVIDER:model_name for backend
      const modelName = `${provider.toUpperCase()}:${model}`;
      const { promise } = api.fun.trip(trimmedCity, days.toString(), modelName);
      const res = await promise;
      setCityResult(res.plan);
      setCityError(null); // Clear error on success
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      // Set user-friendly error message
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCityError(errorMessage);
    } finally {
      setCityLoading(false);
    }
  };

  const handleCravings = async () => {
    if (!model) return;
    setCravingLoading(true);
    setCravingResult('');
    setCravingError(null); // Clear previous errors
    try {
      // Format model name as PROVIDER:model_name for backend
      const modelName = `${provider.toUpperCase()}:${model}`;
      const { promise } = api.fun.cravings('', craving, modelName);
      const res = await promise;
      setCravingResult(res.recommendation);
      setCravingError(null); // Clear error on success
    } catch (e) {
      // Ignore abort errors (request cancelled)
      if (e instanceof Error && (e as any).aborted) {
        return;
      }
      // Set user-friendly error message
      const errorMessage = e instanceof Error ? e.message : String(e);
      setCravingError(errorMessage);
    } finally {
      setCravingLoading(false);
    }
  };

  return (
    <div className="fun-tools">
      <div style={{ marginBottom: '32px', textAlign: 'center' }}>
        <div className="pill-tabs" style={{ display: 'inline-flex' }}>
          <button className={`pill-btn ${subTab === 'city' ? 'active' : ''}`} onClick={() => setSubTab('city')}>
            ✈️ City Planner
          </button>
          <button className={`pill-btn ${subTab === 'cravings' ? 'active' : ''}`} onClick={() => setSubTab('cravings')}>
            🍽️ Cravings
          </button>
        </div>
      </div>

      {subTab === 'city' && (
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'nowrap' }}>
            <h2 className="card-title" style={{ margin: 0 }}>✈️ Trip Planner</h2>
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
            marginBottom: '24px',
            padding: '20px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: 'var(--border-radius-md)',
            border: '1px solid var(--border-subtle)'
          }}>
            <p style={{ 
              color: 'var(--text-secondary)', 
              fontSize: '14px', 
              margin: 0,
              lineHeight: 1.6
            }}>
              🌍 Get a personalized day-by-day itinerary with attractions, food recommendations, and weather forecasts for your destination.
            </p>
          </div>

          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '16px', 
            marginBottom: '20px' 
          }}>
            <div>
              <label style={{ 
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '12px', 
                fontWeight: 600,
                marginBottom: '10px',
                color: 'var(--text-primary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                <span style={{ fontSize: '16px' }}>📍</span>
                <span>City Name</span>
              </label>
              <input 
                type="text" 
                value={city} 
                onChange={e => {
                  setCity(e.target.value);
                  setCityError(null); // Clear error when input changes
                }} 
                placeholder="e.g., Tokyo, Paris, New York..."
                style={{
                  fontSize: '16px',
                  fontWeight: 500
                }}
              />
            </div>
            <div>
              <label style={{ 
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '12px', 
                fontWeight: 600,
                marginBottom: '10px',
                color: 'var(--text-primary)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                <span style={{ fontSize: '16px' }}>📅</span>
                <span>Days</span>
              </label>
              <input 
                type="number" 
                min="1" 
                max="10"
                value={days} 
                onChange={e => {
                  setDays(parseInt(e.target.value));
                  setCityError(null); // Clear error when input changes
                }} 
                style={{
                  fontSize: '16px',
                  fontWeight: 500
                }}
              />
            </div>
          </div>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', 
            gap: '8px', 
            marginBottom: '20px' 
          }}>
            {cityExamples.map(ex => (
              <button
                key={ex}
                onClick={() => {
                  setCity(ex);
                  setCityError(null); // Clear error when selecting example
                }}
                className="btn btn-secondary"
                style={{ fontSize: '13px' }}
              >
                {ex}
              </button>
            ))}
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={handleCityPlan} 
            disabled={cityLoading}
            style={{ width: '100%', fontSize: '16px', padding: '16px' }}
          >
            {cityLoading ? (
              <>
                <div className="loader"></div>
                <span style={{ marginLeft: '12px' }}>Creating Your Itinerary...</span>
              </>
            ) : '✈️ Generate Trip Plan'}
          </button>

          {cityError && (
            <div style={{ 
              marginTop: '20px',
              padding: '16px 20px',
              background: 'rgba(220, 38, 38, 0.1)',
              borderRadius: 'var(--border-radius-md)',
              border: '1px solid rgba(220, 38, 38, 0.3)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px'
            }}>
              <span style={{ fontSize: '18px', flexShrink: 0 }}>⚠️</span>
              <div style={{ flex: 1 }}>
                <p style={{ 
                  margin: 0, 
                  color: 'rgba(248, 113, 113, 1)',
                  fontSize: '14px',
                  lineHeight: '1.5'
                }}>
                  {cityError}
                </p>
              </div>
            </div>
          )}

          {cityResult && (
            <div style={{ 
              marginTop: '32px',
              padding: '28px',
              background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(22, 25, 32, 0.8) 100%)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid rgba(212, 175, 55, 0.2)',
              boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
            }}>
              <h3 style={{ 
                fontFamily: 'var(--font-display)', 
                color: 'var(--accent-gold)', 
                marginTop: 0,
                fontSize: '22px',
                marginBottom: '20px'
              }}>
                Your {city} Adventure
              </h3>
              <div className="prose" style={{ whiteSpace: 'pre-wrap' }}>
                {cityResult}
              </div>
            </div>
          )}
        </div>
      )}

      {subTab === 'cravings' && (
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'nowrap' }}>
            <h2 className="card-title" style={{ margin: 0 }}>🍽️ Cravings Generator</h2>
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
            marginBottom: '24px',
            padding: '20px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: 'var(--border-radius-md)',
            border: '1px solid var(--border-subtle)'
          }}>
            <p style={{ 
              color: 'var(--text-secondary)', 
              fontSize: '14px', 
              margin: 0,
              lineHeight: 1.6
            }}>
              🎲 Can't decide what to eat? Tell us your preferences and we'll suggest something delicious!
            </p>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ 
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '12px', 
              fontWeight: 600,
              marginBottom: '10px',
              color: 'var(--text-primary)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              <span style={{ fontSize: '16px' }}>🍴</span>
              <span>What Are You Craving?</span>
            </label>
            <textarea 
              value={craving} 
              onChange={e => {
                setCraving(e.target.value);
                setCravingError(null); // Clear error when input changes
              }} 
              rows={4} 
              placeholder="Describe your cravings... cuisine type, dietary needs, mood, spice level, etc."
              style={{
                fontSize: '15px',
                lineHeight: '1.6'
              }}
            />
            <p style={{
              fontSize: '12px',
              color: 'var(--text-muted)',
              marginTop: '8px',
              fontStyle: 'italic'
            }}>
              💡 Example: "Something spicy and Asian, vegetarian-friendly" or just type "idk" to be surprised!
            </p>
          </div>

          <button 
            className="btn btn-primary" 
            onClick={handleCravings} 
            disabled={cravingLoading}
            style={{ width: '100%', fontSize: '16px', padding: '16px' }}
          >
            {cravingLoading ? (
              <>
                <div className="loader"></div>
                <span style={{ marginLeft: '12px' }}>Finding Your Perfect Meal...</span>
              </>
            ) : '🎲 Cook Up a Suggestion'}
          </button>

          {cravingError && (
            <div style={{ 
              marginTop: '20px',
              padding: '16px 20px',
              background: 'rgba(220, 38, 38, 0.1)',
              borderRadius: 'var(--border-radius-md)',
              border: '1px solid rgba(220, 38, 38, 0.3)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px'
            }}>
              <span style={{ fontSize: '18px', flexShrink: 0 }}>⚠️</span>
              <div style={{ flex: 1 }}>
                <p style={{ 
                  margin: 0, 
                  color: 'rgba(248, 113, 113, 1)',
                  fontSize: '14px',
                  lineHeight: '1.5'
                }}>
                  {cravingError}
                </p>
              </div>
            </div>
          )}

          {cravingResult && (
            <div style={{ 
              marginTop: '32px',
              padding: '28px',
              background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(22, 25, 32, 0.8) 100%)',
              borderRadius: 'var(--border-radius-lg)',
              border: '1px solid rgba(212, 175, 55, 0.2)',
              boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
            }}>
              <h3 style={{ 
                fontFamily: 'var(--font-display)', 
                color: 'var(--accent-gold)', 
                marginTop: 0,
                fontSize: '22px',
                marginBottom: '20px'
              }}>
                We Recommend
              </h3>
              <div className="prose" style={{ whiteSpace: 'pre-wrap' }}>
                {cravingResult}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
