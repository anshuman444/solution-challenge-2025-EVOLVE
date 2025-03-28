import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { createChatInstance } from './config/gemini';
import type { Message, ChatState, Language } from './types';

function App() {
  const [language, setLanguage] = useState<Language>('en');
  const [input, setInput] = useState('');
  const [chatState, setChatState] = useState<ChatState>({
    messages: [
      { 
        role: 'assistant', 
        content: language === 'en' 
          ? "[English Response]\n1. Welcome to Kisan Mitra, your farming assistant\n2. I can help you with agricultural queries\n3. Ask me about crops, soil, irrigation, or any farming topic\n\n[हिंदी में जवाब]\n1. किसान मित्र में आपका स्वागत है\n2. मैं आपकी कृषि संबंधी जिज्ञासाओं में मदद कर सकता हूं\n3. फसलों, मिट्टी, सिंचाई या किसी भी कृषि विषय के बारे में पूछें"
          : "[English Response]\n1. Welcome to Kisan Mitra, your farming assistant\n2. I can help you with agricultural queries\n3. Ask me about crops, soil, irrigation, or any farming topic\n\n[हिंदी में जवाब]\n1. किसान मित्र में आपका स्वागत है\n2. मैं आपकी कृषि संबंधी जिज्ञासाओं में मदद कर सकता हूं\n3. फसलों, मिट्टी, सिंचाई या किसी भी कृषि विषय के बारे में पूछें"
      }
    ],
    isLoading: false,
    error: null
  });
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatInstanceRef = useRef<any>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatState.messages]);

  useEffect(() => {
    let mounted = true;

    const initializeChat = async () => {
      if (!mounted) return;

      setChatState(prev => ({ ...prev, isLoading: true, error: null }));
      
      try {
        const chat = await createChatInstance(language);
        
        if (!mounted) return;
        
        chatInstanceRef.current = chat;
        setChatState(prev => ({
          ...prev,
          isLoading: false,
          messages: [
            {
              role: 'assistant',
              content: language === 'en'
                ? "[English Response]\n1. Welcome to Kisan Mitra, your farming assistant\n2. I can help you with agricultural queries\n3. Ask me about crops, soil, irrigation, or any farming topic\n\n[हिंदी में जवाब]\n1. किसान मित्र में आपका स्वागत है\n2. मैं आपकी कृषि संबंधी जिज्ञासाओं में मदद कर सकता हूं\n3. फसलों, मिट्टी, सिंचाई या किसी भी कृषि विषय के बारे में पूछें"
                : "[English Response]\n1. Welcome to Kisan Mitra, your farming assistant\n2. I can help you with agricultural queries\n3. Ask me about crops, soil, irrigation, or any farming topic\n\n[हिंदी में जवाब]\n1. किसान मित्र में आपका स्वागत है\n2. मैं आपकी कृषि संबंधी जिज्ञासाओं में मदद कर सकता हूं\n3. फसलों, मिट्टी, सिंचाई या किसी भी कृषि विषय के बारे में पूछें"
            },
            ...prev.messages.slice(1)
          ]
        }));
      } catch (error) {
        if (!mounted) return;
        
        console.error('Chat initialization error:', error);
        setChatState(prev => ({
          ...prev,
          isLoading: false,
          error: language === 'en'
            ? 'Failed to initialize chat. Please check your internet connection and try again.'
            : 'चैट शुरू करने में विफल। कृपया अपना इंटरनेट कनेक्शन जांचें और पुनः प्रयास करें।'
        }));
      }
    };

    initializeChat();

    return () => {
      mounted = false;
    };
  }, [language]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isSubmitting || !chatInstanceRef.current) return;

    const userMessage = input.trim();
    setInput('');
    setIsSubmitting(true);
    
    setChatState(prev => ({
      ...prev,
      messages: [...prev.messages, { role: 'user', content: userMessage }],
      isLoading: true,
      error: null
    }));

    try {
      const result = await chatInstanceRef.current.sendMessage(userMessage);
      const response = await result.response;
      const text = await response.text();
      
      setChatState(prev => ({
        ...prev,
        messages: [...prev.messages, { role: 'assistant', content: text }],
        isLoading: false
      }));
    } catch (error) {
      console.error('Chat error:', error);
      setChatState(prev => ({
        ...prev,
        isLoading: false,
        error: language === 'en' 
          ? 'Failed to get response. Please try again.'
          : 'प्रतिक्रिया प्राप्त करने में विफल। कृपया पुनः प्रयास करें।'
      }));
    } finally {
      setIsSubmitting(false);
    }
  };

  const placeholderText = language === 'en' 
    ? "Ask your farming related question..."
    : "अपना कृषि संबंधित प्रश्न पूछें...";

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white">
      <div className="max-w-4xl mx-auto px-4 flex flex-col min-h-screen">
        {/* Header */}
        <div className="py-4 sm:py-8 text-center">
          <h1 className="text-2xl sm:text-4xl font-bold text-green-700 mb-4">
            Kisan Mitra | किसान मित्र
          </h1>
          <div className="flex justify-center gap-2 sm:gap-4">
            <button
              onClick={() => setLanguage('en')}
              className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg border-2 transition-colors ${
                language === 'en'
                  ? 'border-green-600 bg-green-600 text-white'
                  : 'border-green-200 text-green-700 hover:border-green-400'
              }`}
            >
              English
            </button>
            <button
              onClick={() => setLanguage('hi')}
              className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg border-2 transition-colors ${
                language === 'hi'
                  ? 'border-green-600 bg-green-600 text-white'
                  : 'border-green-200 text-green-700 hover:border-green-400'
              }`}
            >
              हिंदी
            </button>
          </div>
        </div>
        
        {/* Chat container */}
        <div className="flex-1 bg-white rounded-xl shadow-lg border border-green-100 overflow-hidden mb-4 sm:mb-6 flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-gradient-to-b from-green-50 to-white">
            <div className="flex flex-col gap-4 sm:gap-6">
              {chatState.messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))}
              {chatState.isLoading && (
                <div className="flex justify-center">
                  <Loader2 className="w-6 h-6 text-green-600 animate-spin" />
                </div>
              )}
              {chatState.error && (
                <div className="text-red-500 text-center p-4 bg-red-50 rounded-lg">
                  {chatState.error}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          </div>

          <form 
            onSubmit={handleSubmit} 
            className="p-4 border-t border-green-100 bg-white"
          >
            <div className="flex gap-2 sm:gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={placeholderText}
                className="flex-1 px-3 sm:px-4 py-2 sm:py-3 rounded-lg border-2 border-green-200 focus:outline-none focus:border-green-500 transition-colors text-base"
                disabled={isSubmitting}
              />
              <button
                type="submit"
                disabled={isSubmitting || !chatInstanceRef.current}
                className="px-4 sm:px-6 py-2 sm:py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:hover:bg-green-600 transition-colors flex items-center gap-2 active:bg-green-800 touch-manipulation"
                style={{ WebkitTapHighlightColor: 'transparent' }}
              >
                {isSubmitting ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;