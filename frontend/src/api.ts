const API_BASE = '/api';

/**
 * Default timeout for API calls in milliseconds (30 seconds)
 */
const DEFAULT_TIMEOUT = 30000;

/**
 * Result type for cancellable API calls - includes both the promise and a cancel function
 */
export interface CancellableApiCall<T> {
  promise: Promise<T>;
  cancel: () => void;
  controller: AbortController;
}

/**
 * Typed helper function for API calls with timeout and cancellation support.
 * Creates an AbortController, sets a default timeout (30s) that aborts the request,
 * passes controller.signal into fetch, clears the timeout in a finally block,
 * and rejects on non-ok responses.
 * 
 * Returns both the promise and a cancel function so callers can cancel on unmount/navigation.
 * 
 * @param url - The API endpoint URL
 * @param options - Optional fetch options (signal will be merged with AbortController signal)
 * @param timeoutMs - Optional timeout in milliseconds (defaults to DEFAULT_TIMEOUT)
 * @returns Object containing promise, cancel function, and AbortController
 */
function apiCall<T>(
  url: string, 
  options?: RequestInit, 
  timeoutMs: number = DEFAULT_TIMEOUT
): CancellableApiCall<T> {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  
  // Merge the provided signal with our controller signal
  // If a signal is provided, abort our controller when that signal aborts
  if (options?.signal) {
    if (options.signal.aborted) {
      controller.abort();
    } else {
      options.signal.addEventListener('abort', () => controller.abort());
    }
  }
  
  const signal = controller.signal;
  
  // Set up timeout that will abort the request
  timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);
  
  const promise = (async (): Promise<T> => {
    try {
      const response = await fetch(url, {
        ...options,
        signal,
      });
      
      // Clear timeout since we got a response
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      
      // For non-ok responses, attempt to extract error message from response body
      if (!response.ok) {
        let errorMessage = `API call failed: ${response.status} ${response.statusText}`;
        
        try {
          // Try to parse JSON error response
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorData = await response.json();
            // Extract error message from common error response formats
            if (errorData.error) {
              errorMessage = typeof errorData.error === 'string' 
                ? errorData.error 
                : JSON.stringify(errorData.error);
            } else if (errorData.message) {
              errorMessage = errorData.message;
            } else if (errorData.detail) {
              errorMessage = errorData.detail;
            } else {
              // Fallback: stringify the entire error object if no standard field found
              errorMessage = JSON.stringify(errorData);
            }
          } else {
            // For non-JSON responses, try to get text
            try {
              const text = await response.text();
              if (text) {
                errorMessage = text;
              }
            } catch {
              // If text parsing also fails, use the default message
            }
          }
        } catch (parseError) {
          // If JSON parsing fails, fall back to safe default message
          errorMessage = `API call failed: ${response.status} ${response.statusText}`;
        }
        
        const error = new Error(errorMessage);
        (error as any).status = response.status;
        (error as any).statusText = response.statusText;
        throw error;
      }
      
      // For successful responses, attempt to parse JSON with error handling
      try {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          return await response.json() as T;
        } else {
          // If response is not JSON, return empty object or throw
          // Some endpoints might return non-JSON (e.g., file downloads)
          // For now, we'll try to parse anyway and let it fail naturally
          return await response.json() as T;
        }
      } catch (parseError) {
        // If JSON parsing fails for successful response, throw descriptive error
        throw new Error(
          `Failed to parse JSON response from ${url}: ${parseError instanceof Error ? parseError.message : 'Unknown error'}`
        );
      }
    } catch (error) {
      // Handle abort errors specifically
      if (error instanceof Error && error.name === 'AbortError') {
        const abortError = new Error('Request was cancelled or timed out');
        (abortError as any).aborted = true;
        (abortError as any).timeout = timeoutId === null; // If timeoutId is null, it was cleared, so it was a manual cancel
        throw abortError;
      }
      throw error;
    } finally {
      // Always clear timeout in finally block
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    }
  })();
  
  return {
    promise,
    cancel: () => {
      controller.abort();
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    },
    controller,
  };
}

// Response type definitions

export interface HealthResponse {
  status: string;
}

export interface ModelListResponse {
  models: string[];
}

export interface AnalyzeResponse {
  message: string;
  summary: string | null;
  example_queries: string | null;
  video_title?: string | null;
  article_title?: string | null;
  media_title?: string | null;
  file_title?: string | null;
  video_memoryupload_status?: string;
  article_memoryupload_status?: string;
  media_memoryupload_status?: string;
  file_memoryupload_status?: string;
}

export interface DocQAResponse {
  answer: string;
}

export interface DocQAResetResponse {
  status: string;
}

export interface ChatResponse {
  response: string;
}

export interface TripResponse {
  plan: string;
}

export interface CravingsResponse {
  recommendation: string;
}

export interface ImageGenerateResponse {
  image_path?: string;
  error?: string;
}

export interface ImageEditResponse {
  image_path?: string;
  error?: string;
}

export interface ImageEnhanceResponse {
  enhanced_prompt: string;
}

export interface ImageSurpriseResponse {
  prompt: string;
}

export interface ImageUploadResponse {
  file_path: string;
}

export const api = {
  health: (): CancellableApiCall<HealthResponse> => 
    apiCall<HealthResponse>(`${API_BASE}/health`),
  
  models: {
    list: (provider: string): CancellableApiCall<ModelListResponse> => 
      apiCall<ModelListResponse>(`${API_BASE}/models/${provider}`),
  },
  
  analyze: {
    file: (formData: FormData): CancellableApiCall<AnalyzeResponse> => 
      apiCall<AnalyzeResponse>(`${API_BASE}/analyze/file`, {
        method: 'POST',
        body: formData
      }),
    
    youtube: (url: string, memorize: boolean, model: string): CancellableApiCall<AnalyzeResponse> => 
      apiCall<AnalyzeResponse>(`${API_BASE}/analyze/youtube`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, memorize, model_name: model })
      }),
    
    article: (url: string, memorize: boolean, model: string): CancellableApiCall<AnalyzeResponse> => 
      apiCall<AnalyzeResponse>(`${API_BASE}/analyze/article`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, memorize, model_name: model })
      }),
    
    media: (url: string, memorize: boolean, model: string): CancellableApiCall<AnalyzeResponse> => 
      apiCall<AnalyzeResponse>(`${API_BASE}/analyze/media`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, memorize, model_name: model })
      }),
  },
  
  docqa: {
    ask: (message: string, history: string[][], model: string): CancellableApiCall<DocQAResponse> => 
      apiCall<DocQAResponse>(`${API_BASE}/docqa/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history, model_name: model })
      }),
    
    reset: (): CancellableApiCall<DocQAResetResponse> => 
      apiCall<DocQAResetResponse>(`${API_BASE}/docqa/reset`, { method: 'POST' }),
  },
  
  chat: {
    internet: (message: string, history: string[][], model: string, max_tokens: number, temperature: number): CancellableApiCall<ChatResponse> => 
      apiCall<ChatResponse>(`${API_BASE}/chat/internet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history, model_name: model, max_tokens, temperature })
      }),
  },
  
  fun: {
    trip: (city: string, days: string, model: string): CancellableApiCall<TripResponse> => 
      apiCall<TripResponse>(`${API_BASE}/fun/trip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city, days, model_name: model })
      }),
    
    cravings: (city: string, cuisine: string, model: string): CancellableApiCall<CravingsResponse> => 
      apiCall<CravingsResponse>(`${API_BASE}/fun/cravings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city, cuisine, model_name: model })
      }),
  },
  
  image: {
    generate: (prompt: string, enhanced_prompt: string, size: string, provider: string): CancellableApiCall<ImageGenerateResponse> => 
      apiCall<ImageGenerateResponse>(`${API_BASE}/image/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, enhanced_prompt, size, provider })
      }),
      
    edit: (img_path: string, prompt: string, enhanced_prompt: string, size: string, provider: string): CancellableApiCall<ImageEditResponse> => 
      apiCall<ImageEditResponse>(`${API_BASE}/image/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ img_path, prompt, enhanced_prompt, size, provider })
      }),
      
    enhance: (prompt: string, provider: string): CancellableApiCall<ImageEnhanceResponse> => 
      apiCall<ImageEnhanceResponse>(`${API_BASE}/image/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, provider })
      }),
    
    surprise: (provider: string): CancellableApiCall<ImageSurpriseResponse> => 
      apiCall<ImageSurpriseResponse>(`${API_BASE}/image/surprise`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider })
      }),
    
    upload: (file: File): CancellableApiCall<ImageUploadResponse> => {
      const fd = new FormData();
      fd.append('file', file);
      return apiCall<ImageUploadResponse>(`${API_BASE}/image/upload`, {
        method: 'POST',
        body: fd
      });
    }
  }
};

